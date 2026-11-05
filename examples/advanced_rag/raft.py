"""RAFT, Retrieval-Augmented Fine-Tuning (Zhang et al., Berkeley, 2024): build fine-tuning
examples of (question, oracle doc + distractor docs) -> chain-of-thought answer that quotes
the oracle. In a fraction (1 - p) of examples the oracle is REMOVED, teaching the model to
ignore distractors and not over-trust retrieval. Output is a chat-format JSONL dataset."""

import json
import random

from rag_kit import CORPUS, get_llm, parse_json, show

QGEN = """<text>{text}</text>
Write 2 questions answerable from this text. Return a JSON list of strings."""
COT = """<context>{ctx}</context>
<question>{q}</question>
Answer with reasoning. Quote the supporting sentence between ##begin_quote## and
##end_quote##, then give the final answer after <ANSWER>:."""


def make_raft_dataset(p_oracle: float = 0.8, n_distractors: int = 3, seed: int = 0):
    rng, llm, rows = random.Random(seed), get_llm(), []
    for oracle in CORPUS:
        for q in parse_json(llm(QGEN.format(text=oracle.text), task="questions"), []):
            distractors = rng.sample([d for d in CORPUS if d.id != oracle.id], n_distractors)
            docs = distractors + ([oracle] if rng.random() < p_oracle else
                                  rng.sample([d for d in CORPUS if d not in distractors], 1))
            rng.shuffle(docs)
            ctx = "\n".join(f"<document>{d.text}</document>" for d in docs)
            # The teacher sees the oracle so the target answer is correct even when the
            # student's context does not contain it.
            teacher_ctx = f"<document>{oracle.text}</document>"
            target = llm(COT.format(ctx=teacher_ctx, q=q), task="answer")
            rows.append({"messages": [
                {"role": "user", "content": f"{ctx}\n<question>{q}</question>"},
                {"role": "assistant", "content": target}],
                "oracle_in_context": oracle in docs})
    return rows


def demo() -> None:
    rows = make_raft_dataset()
    show("RAFT examples", {"n": len(rows),
                           "oracle_share": round(sum(r["oracle_in_context"] for r in rows)
                                                 / max(len(rows), 1), 2)})
    print(json.dumps(rows[0], indent=1)[:600])


if __name__ == "__main__":
    demo()
