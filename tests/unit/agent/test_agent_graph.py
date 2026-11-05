import numpy as np
from agent_fakes import make_chunk

from chatbot.agent import GraphRetriever, HeuristicEntityExtractor, personalized_pagerank
from chatbot.core.types import ACL, Principal

ALICE = Principal(user_id="alice", groups=frozenset({"staff"}))


def _chunks(hidden_bridge: bool = False):
    return [
        make_chunk("c1", "d1", "Alice Smith works at Acme Corp as an engineer."),
        make_chunk(
            "c2",
            "d2",
            "Acme Corp is headquartered in Berlin.",
            acl=ACL(allow_groups=["execs"]) if hidden_bridge else None,
        ),
        make_chunk("c3", "d3", "Berlin hosts the annual Tech Summit."),
        make_chunk("c4", "d4", "Bob Jones enjoys hiking in Patagonia."),
    ]


def test_heuristic_entities():
    ents = HeuristicEntityExtractor().extract_one(
        "The Refund Policy applies to Acme Corp and SKU-1234 in the EU."
    )
    assert {"refund policy", "acme corp", "sku 1234", "eu"} <= ents
    assert "the" not in ents


def test_ppr_simple_chain():
    # nodes 0-1-2 chain (undirected), seed on 0 → scores decrease along the chain
    src = np.array([0, 1, 1, 2])
    dst = np.array([1, 0, 2, 1])
    p = personalized_pagerank(3, src, dst, np.ones(4), np.array([1.0, 0, 0]))
    assert p.sum() == np.float64(p.sum()) and abs(p.sum() - 1) < 1e-9
    assert p[0] > p[1] > p[2] > 0


async def test_ppr_ranks_connected_chunk():
    g = GraphRetriever()
    await g.add(_chunks())
    res = await g.retrieve("Where is the company Alice Smith works for located?", 4, ALICE)
    ids = [r.chunk.chunk_id for r in res]
    assert ids[0] == "c1"
    assert "c2" in ids  # multi-hop: connected through "Acme Corp"
    assert "c4" not in ids  # disconnected
    assert ids.index("c2") < ids.index("c3") if "c3" in ids else True


async def test_graph_respects_acl():
    g = GraphRetriever()
    await g.add(_chunks(hidden_bridge=True))
    res = await g.retrieve("Where is the company Alice Smith works for located?", 4, ALICE)
    ids = [r.chunk.chunk_id for r in res]
    assert "c2" not in ids and "c3" not in ids  # hidden chunk is neither returned nor a bridge
    exec_user = Principal(user_id="eve", groups=frozenset({"execs"}))
    assert "c2" in [
        r.chunk.chunk_id for r in await g.retrieve("Alice Smith employer", 4, exec_user)
    ]
    assert await g.delete_doc("d1") == 1
    assert await g.retrieve("Alice Smith", 4, ALICE) == []
