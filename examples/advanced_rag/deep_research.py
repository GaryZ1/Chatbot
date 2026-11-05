"""Deep research / multi-agent RAG (the OpenAI, Gemini and Claude "Research" pattern;
STORM, Shao et al., 2024): a planner splits the question into research threads, parallel
researcher agents each run their own search loop with a separate context window, and a
synthesiser writes a cited report from their findings. Minutes per query; route to it only
for open-ended, multi-faceted questions."""

from concurrent.futures import ThreadPoolExecutor

from agentic_react import agent_loop
from rag_kit import get_llm, parse_json, show

PLAN = """<question>{q}</question>
Plan the research: list 2-4 independent sub-questions that together cover the question.
Return a JSON list of strings."""
SYNTH = """<question>{q}</question>
<findings>
{findings}
</findings>
Write a short, well-structured report that answers the question from the findings. Keep
every citation. Point out gaps or conflicts between findings."""


def deep_research(question: str, max_threads: int = 4) -> dict:
    llm = get_llm()
    threads = parse_json(llm(PLAN.format(q=question), task="plan"), [question])[:max_threads]
    with ThreadPoolExecutor(max_workers=max_threads) as pool:     # isolated sub-agents
        results = list(pool.map(lambda sq: agent_loop(sq, max_steps=4), threads))
    findings = "\n".join(f'<source id="{i + 1}" title="{t}">{r["answer"]}</source>'
                         for i, (t, r) in enumerate(zip(threads, results, strict=True)))
    report = llm(SYNTH.format(q=question, findings=findings), task="synthesize")
    return {"threads": threads, "findings": [r["answer"] for r in results], "report": report}


def demo() -> None:
    show("Deep research", deep_research("Who leads Orbital Systems and what has the company "
                                        "acquired?"))


if __name__ == "__main__":
    demo()
