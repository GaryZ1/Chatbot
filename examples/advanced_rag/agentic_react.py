"""Agentic RAG with a ReAct loop (Yao et al., 2022): the LLM plans, calls retrieval tools,
observes results and repeats until it can answer. Production guards: step budget, loop
detection, tool allow-list, every call logged. Two implementations:
  * ``agent_loop``: a provider-neutral JSON action protocol (works with the offline LLM);
  * ``claude_tool_agent``: native Claude tool use with the Anthropic SDK."""

import json
from collections.abc import Callable

from rag_kit import CORPUS, Retriever, get_llm, parse_json, show

Tool = tuple[str, Callable[[str], str]]           # (description, function(input) -> text)

AGENT_PROMPT = """You answer questions about a company knowledge base by using tools.
Tools:
{tools}
<question>{q}</question>
{transcript}
Think, then return JSON {{"thought": "...", "action": "<tool name or finish>",
"input": "<tool input, or the final answer with citations if action is finish>"}}."""


def default_tools() -> dict[str, Tool]:
    r, docs = Retriever(), {d.id: d for d in CORPUS}
    return {
        "search": ("search(query): hybrid search, returns the top passages with ids",
                   lambda q: "\n".join(f"[{c.id}] {c.text}" for c in r.search(q, k=3))),
        "read": ("read(doc_id): the full text of one document, e.g. read(d03)",
                 lambda d: docs[d.split("#")[0]].text if d.split("#")[0] in docs
                 else "error: unknown doc id"),
    }


def agent_loop(question: str, tools: dict[str, Tool] | None = None,
               max_steps: int = 6) -> dict:
    llm, tools = get_llm(), tools or default_tools()
    spec = "\n".join(f"- {name}: {desc}" for name, (desc, _) in tools.items())
    transcript, trace, seen = [], [], set()
    for _ in range(max_steps):
        out = parse_json(llm(AGENT_PROMPT.format(tools=spec, q=question,
                                                 transcript="\n".join(transcript)),
                             task="agent"), {})
        action, arg = out.get("action", "finish"), str(out.get("input", ""))
        trace.append({"thought": out.get("thought"), "action": action, "input": arg})
        if action == "finish":
            return {"answer": arg, "trace": trace}
        if action not in tools:                                # allow-list
            obs = f"error: unknown tool {action!r}"
        elif (action, arg) in seen:                            # loop detection
            obs = "error: you already ran this exact call; try something different"
        else:
            obs = tools[action][1](arg)
        seen.add((action, arg))
        transcript.append(f"<action>{action}({arg})</action>\n<observation>{obs}</observation>")
    final = llm(f"<context>{' '.join(transcript)}</context>\n<question>{question}</question>",
                task="answer")                                 # budget exhausted: best effort
    return {"answer": final, "trace": trace, "budget_exhausted": True}


def claude_tool_agent(question: str, max_turns: int = 8) -> str:
    """Native tool use: Claude emits tool_use blocks; we run them and return tool_result
    blocks (all results of one turn in ONE user message), until stop_reason != tool_use."""
    import anthropic

    client, tools = anthropic.Anthropic(), default_tools()
    schema = [{"name": n, "description": d, "strict": True,
               "input_schema": {"type": "object", "additionalProperties": False,
                                "properties": {"input": {"type": "string"}},
                                "required": ["input"]}} for n, (d, _) in tools.items()]
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        resp = client.beta.messages.create(
            model="claude-opus-5-5", max_tokens=16000, tools=schema, messages=messages,
            system="Answer from the knowledge base only, using the tools. Cite passage ids.",
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                fn = tools[block.name][1]
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": fn(block.input["input"])})
        messages.append({"role": "user", "content": results})
    return "Stopped: turn budget exhausted."


def demo() -> None:
    result = agent_loop("Where did the CTO of Orbital Systems work before?")
    show("Agent trace", result["trace"])
    show("Answer", result["answer"])
    print(json.dumps({"steps": len(result["trace"])}))


if __name__ == "__main__":
    demo()
