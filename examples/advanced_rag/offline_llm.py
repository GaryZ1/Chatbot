"""Deterministic offline stand-in for an LLM, used by every example when RAG_LLM is not
"claude". It imitates the *shape* of each answer (JSON lists, grades, actions, citations)
with lexical heuristics keyed by the ``task`` hint, so the control flow of every method can
run in CI with no network. It is a test double: never draw quality conclusions from it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rag_kit import (
    content_terms,
    entities,
    extract_all,
    extract_tag,
    overlap,
    split_sentences,
    tokenize,
)


class OfflineLLM:
    """Deterministic heuristics keyed by ``task``. Lets every example run offline and in CI.

    It imitates the *shape* of each answer (JSON lists, grades, actions) with lexical
    rules. Quality conclusions must come from a real model plus an evaluation set.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, prompt: str, *, task: str = "answer", system: str = "",
                 max_tokens: int = 16000) -> str:
        self.calls.append((task, prompt))
        handler = getattr(self, f"_t_{task}", self._t_answer)
        return handler(prompt)

    # --- helpers
    @staticmethod
    def _q(p: str) -> str:
        return extract_tag(p, "question") or extract_tag(p, "query") or p.strip()

    @staticmethod
    def _sentences(p: str) -> list[tuple[str, str]]:
        """(source_id, sentence) pairs from <source id=..> or <document> blocks."""
        out: list[tuple[str, str]] = []
        for sid, body in re.findall(r'<source id="([^"]+)"[^>]*>(.*?)</source>', p, re.S):
            out += [(sid, s) for s in split_sentences(body)]
        for body in extract_all(p, "observation") + extract_all(p, "information"):
            for line in body.splitlines():
                if m := re.match(r"\s*\[([^\]]+)\]\s*(.+)", line):
                    out += [(m.group(1), s) for s in split_sentences(m.group(2))]
        if not out:
            for tag in ("document", "context", "memory", "information", "observation"):
                for body in extract_all(p, tag):
                    out += [("", s) for s in split_sentences(body)]
        return out

    def _best(self, q: str, p: str) -> tuple[str, str, float]:
        best = ("", "", 0.0)
        for sid, s in self._sentences(p):
            sc = overlap(q, s) + 0.01 * overlap(s, q)
            if sc > best[2]:
                best = (sid, s, sc)
        return best

    # --- task handlers
    def _t_answer(self, p: str) -> str:
        sid, sent, sc = self._best(self._q(p), p)
        if sc < 0.4:
            return "I don't know based on the provided sources."
        return f"{sent} [{sid}]" if sid else sent

    _t_reduce = _t_answer

    def _t_draft(self, p: str) -> str:
        sid, sent, sc = self._best(self._q(p), p)
        return json.dumps({"draft": f"{sent} [{sid}]" if sc >= 0.2 else "I don't know.",
                           "rationale": f"Source {sid} states: {sent}" if sc >= 0.2 else ""})

    def _t_verify(self, p: str) -> str:
        sc = overlap(self._q(p), extract_tag(p, "answer") + " " + extract_tag(p, "rationale"))
        return json.dumps({"verdict": "yes" if sc >= 0.5 else "no", "score": round(sc, 2)})

    def _t_sufficient(self, p: str) -> str:
        """Every specific question term must be (softly) present in the context."""
        ctx = set(tokenize(extract_tag(p, "context")))
        terms = content_terms(self._q(p))
        hit = [t for t in terms if any(c.startswith(t[: max(3, len(t) - 2)]) for c in ctx)]
        ok = bool(terms) and len(hit) == len(terms)
        return json.dumps({"verdict": "yes" if ok else "no",
                           "score": round(len(hit) / max(len(terms), 1), 2)})

    def _t_revise(self, p: str) -> str:
        kept = [c for c in extract_all(p, "check") if "don't know" not in c.lower()]
        return " ".join(kept) or "I don't know based on the provided sources."

    def _t_rewrite(self, p: str) -> str:
        return " ".join(content_terms(self._q(p)))

    def _t_hyde(self, p: str) -> str:
        q = self._q(p).rstrip("?")
        return f"{q}. " + " ".join(entities(q))

    def _t_queries(self, p: str) -> str:
        q = self._q(p)
        terms = content_terms(q)
        return json.dumps([q, " ".join(terms), " ".join(entities(q) + terms[-2:])])

    def _t_step_back(self, p: str) -> str:
        q = re.sub(r"\b(19|20)\d\d\b|\b\d+(\.\d+)?\b", "", self._q(p))
        ents = entities(q)
        return f"What is the background of {ents[0]}?" if ents else q

    def _t_decompose(self, p: str) -> str:
        q = self._q(p).rstrip("?")
        parts = [s.strip() for s in re.split(r"\band\b|;|, then", q) if s.strip()]
        if len(parts) < 2:
            ents = entities(q)
            parts = [f"What is known about {e}?" for e in ents[:2]] or [q]
        return json.dumps([x if x.endswith("?") else x + "?" for x in parts])

    def _t_grade(self, p: str) -> str:
        doc = extract_tag(p, "document") or p
        sc = overlap(self._q(p), doc)
        return json.dumps({"relevant": sc >= 0.34, "score": round(sc, 2)})

    def _t_judge(self, p: str) -> str:
        claim = extract_tag(p, "claim") or extract_tag(p, "answer") or self._q(p)
        ctx = extract_tag(p, "context") or p
        sc = overlap(claim, ctx)
        return json.dumps({"verdict": "yes" if sc >= 0.5 else "no", "score": round(sc, 2)})

    def _t_classify(self, p: str) -> str:
        q = self._q(p)
        n_ent, n_words = len(entities(q)), len(tokenize(q))
        if n_words <= 3 and not n_ent:
            label = "A"
        elif n_words >= 14 or " of the " in q.lower() or " that " in q.lower():
            label = "C"
        else:
            label = "B"
        return json.dumps({"label": label})

    def _t_filters(self, p: str) -> str:
        q = self._q(p)
        out: dict[str, Any] = {"query": " ".join(content_terms(re.sub(r"\b20\d\d\b", "", q)))}
        if m := re.search(r"\b(20\d\d)\b", q):
            out["year"] = int(m.group(1))
        for t in ("policy", "product", "security", "finance", "people"):
            if t in q.lower():
                out["type"] = t
        return json.dumps(out)

    def _t_summarize(self, p: str) -> str:
        sents = [s for _, s in self._sentences(p)] or split_sentences(p)
        return " ".join(sents[:: max(1, len(sents) // 3)][:3])

    def _t_context(self, p: str) -> str:
        doc = extract_tag(p, "document")
        title = doc.splitlines()[0] if doc else "a document"
        return f"This chunk is from '{title}', about {', '.join(entities(doc)[:3])}."

    def _t_entities(self, p: str) -> str:
        return json.dumps(entities(extract_tag(p, "text") or self._q(p)))

    def _t_extract_graph(self, p: str) -> str:
        text = extract_tag(p, "text") or p
        ents: dict[str, dict[str, str]] = {}
        rels: list[dict[str, str]] = []
        for s in split_sentences(text):
            es = entities(s)
            for e in es:
                ents.setdefault(e, {"name": e, "type": "entity", "description": s})
            rels += [{"source": a, "target": b, "description": s}
                     for i, a in enumerate(es) for b in es[i + 1 :]]
        return json.dumps({"entities": list(ents.values()), "relations": rels})

    def _t_keywords(self, p: str) -> str:
        q = self._q(p)
        return json.dumps({"low_level": entities(q), "high_level": content_terms(q)[-3:]})

    def _t_propositions(self, p: str) -> str:
        text = extract_tag(p, "text") or p
        subject = (entities(text) or ["It"])[0]
        props = []
        for s in split_sentences(text):
            s = re.sub(r"^(Its|It)\b", subject + ("'s" if s.startswith("Its") else ""), s)
            props += [x.strip() + "." for x in re.split(r";\s+|,\s+and\s+", s.rstrip("."))
                      if len(tokenize(x)) >= 3]
        return json.dumps(props)

    def _t_questions(self, p: str) -> str:
        text = extract_tag(p, "text") or p
        return json.dumps([f"What does the text say about {e}?" for e in entities(text)[:3]])

    def _t_notes(self, p: str) -> str:
        _sid, sent, sc = self._best(self._q(p), p)
        return sent if sc >= 0.2 else "This document is irrelevant to the question."

    def _t_confidence(self, p: str) -> str:
        sentence = extract_tag(p, "sentence")
        ctx = extract_tag(p, "context")
        return json.dumps({"confident": bool(ctx) and overlap(sentence, ctx) >= 0.6})

    def _t_verify_questions(self, p: str) -> str:
        draft = extract_tag(p, "draft")
        draft = re.sub(r"\s*\[[^\]]+\]", "", draft)
        return json.dumps([f"Is it true that {s.rstrip('.')}?" for s in split_sentences(draft)])

    def _t_memory_clues(self, p: str) -> str:
        q = self._q(p)
        _sid, sent, _ = self._best(q, p)
        return json.dumps([q, *entities(q), *([sent] if sent else [])])

    def _t_map(self, p: str) -> str:
        _sid, sent, sc = self._best(self._q(p), p)
        summary = extract_tag(p, "report") or sent
        return json.dumps({"points": [{"text": split_sentences(summary)[0] if summary else "",
                                       "score": round(100 * max(sc, 0.05))}]})

    def _t_next_step(self, p: str) -> str:
        """Iterative retrieval (IRCoT / Self-Ask): emit the next reasoning step or stop."""
        q, steps = self._q(p), extract_all(p, "step")
        sents = [(sid, x) for sid, x in self._sentences(p) if x not in steps]
        focus = q + " " + " ".join(" ".join(entities(x)) for x in steps)
        best = max(sents, key=lambda t: overlap(focus, t[1]), default=None)
        if len(steps) >= 2 or best is None or overlap(focus, best[1]) < 0.15:
            return json.dumps({"done": True, "answer": steps[-1] if steps else
                               "I don't know based on the provided sources."})
        follow = [e for e in entities(best[1]) if e.lower() not in q.lower()] or entities(q)
        return json.dumps({"done": False, "thought": best[1],
                           "query": f"What is known about {follow[0]}?" if follow else q})

    def _t_next_sentence(self, p: str) -> str:
        q, so_far = self._q(p), extract_tag(p, "answer_so_far")
        sents = [x for _, x in self._sentences(p) if x not in so_far]
        best = max(sents, key=lambda x: overlap(q, x), default="")
        if len(split_sentences(so_far)) >= 2 or not best or overlap(q, best) < 0.2:
            return "END"
        return best

    def _t_direct(self, p: str) -> str:
        return "Hello! Ask me anything about Orbital Systems."

    def _t_agent(self, p: str) -> str:
        """ReAct step: search the question, hop once via a bridge entity, then finish."""
        q, obs = self._q(p), extract_all(p, "observation")
        if not obs:
            return json.dumps({"thought": "Search first.", "action": "search", "input": q})
        followed = [a.split("(", 1)[-1].rstrip(")") for a in extract_all(p, "action")[1:]]
        focus = " ".join([q, *followed])
        # after a hop, read only what the newest observation added (the bridge evidence)
        fresh = [ln for ln in obs[-1].splitlines() if not any(ln in o for o in obs[:-1])]
        sid, best, _ = self._best(focus, f"<observation>{chr(10).join(fresh)}</observation>"
                                  if followed and fresh else p)
        hopped = any(f.split(" | ")[0].lower() in best.lower() for f in followed)
        new = [e for e in entities(best) if e.lower() not in focus.lower()]
        if hopped or len(obs) >= 3 or not new:
            ans = f"{best} [{sid}]" if sid else best
            return json.dumps({"thought": "Enough evidence.", "action": "finish", "input": ans})
        return json.dumps({"thought": f"I need to know more about {new[-1]}.",
                           "action": "search", "input": new[-1]})

    def _t_plan(self, p: str) -> str:
        return self._t_decompose(p)

    def _t_sql(self, p: str) -> str:
        q = self._q(p).lower()
        if "highest" in q or "max" in q:
            return "SELECT quarter, revenue_m_eur FROM revenue ORDER BY revenue_m_eur DESC LIMIT 1"
        if "average" in q:
            return "SELECT AVG(revenue_m_eur) FROM revenue"
        return "SELECT quarter, revenue_m_eur, operators FROM revenue ORDER BY quarter LIMIT 10"
