# 11 — Conversation & Memory

## 1. Multi-turn handling

| Technique | What | Pros | Cons |
|---|---|---|---|
| **Full history** | Send all turns | Simple, exact | Token growth; context rot |
| **Sliding window** | Last N turns | Bounded | Forgets early facts |
| **Running summary** | LLM-summarized older turns + recent raw turns | Bounded and informative | Summary drift and loss |
| **Query condensing** | Rewrite each follow-up into a standalone query for retrieval (Module 06) | Correct retrieval in chat | An extra call |
| **Retrieved history** | Embed past turns; retrieve the relevant ones | Scales to long sessions | Complexity |
| **Entity / slot tracking** | Maintain a structured state (product, region, order_id) | Precise follow-ups | Schema design |

**Key insight:** *retrieval* uses the condensed standalone query; *generation* sees the recent turns + summary + retrieved sources.

## 2. Memory tiers
| Tier | Content | Storage | Lifetime |
|---|---|---|---|
| Working | Current turn context | Prompt | One turn |
| Short-term | Session turns, summary, slots | Redis / DB | Session |
| Long-term semantic | User facts and preferences ("prefers Python", "is on the EMEA team") | Vector + KV store | Persistent, editable |
| Episodic | Past conversations and outcomes | Vector store | Persistent, with TTL |
| Procedural | Learned instructions, playbooks | Prompt / config | Versioned |

Frameworks: Mem0, Letta (MemGPT), Zep/Graphiti (temporal knowledge graph memory), LangGraph memory stores. Benchmarks: LoCoMo, LongMemEval.

## 3. Memory rules
- **Write policy:** extract memories with an LLM *selectively* (stable facts and preferences only); dedupe and update instead of appending; timestamp everything.
- **User control:** show, edit, and delete memories (privacy and GDPR).
- **Never store secrets or sensitive PII** unless policy allows it; respect tenant boundaries.
- **Evaluate:** does memory improve task success without injecting stale or wrong facts?

## 4. UX patterns that matter
- Suggested follow-ups; clarifying questions for ambiguity.
- Visible sources with highlighted spans; "open document" links.
- Thumbs up/down + reason codes → feeds evaluation (Module 12).
- Handoff to a human with the conversation summary (support bots).
- Show "searching X…" progress for agentic flows.

## Expert checks
- Why should retrieval use the condensed query while generation sees the raw recent turns?
- How do you stop long-term memory from becoming a source of stale hallucinations?
