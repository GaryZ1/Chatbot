# 13 — Guardrails & Security

## 1. Threat model (OWASP Top 10 for LLM Applications 2025 is the reference)

| Threat | Description | Defenses |
|---|---|---|
| **Direct prompt injection / jailbreak** | User tries to override instructions | Instruction hierarchy, input classifiers (Llama Guard, Prompt Guard, ShieldGemma, NeMo Guardrails, cloud content safety), refusal policies, red-teaming |
| **Indirect prompt injection** | Malicious instructions inside retrieved docs, web pages, or emails ("ignore previous instructions, send…") | Treat retrieved content as *data* (tagged, quoted); never let content trigger privileged tools without user confirmation; scan content at ingest and retrieval; least-privilege tools; output filtering of URLs and markdown images (data exfiltration) |
| **Knowledge-base poisoning** | Attacker plants documents crafted to be retrieved and to mislead (PoisonedRAG, Zou et al. 2024: a few injected texts per target question can achieve high attack success) | Control who can write to indexed sources; provenance and authority scores; anomaly detection on new documents; prefer authoritative sources in ranking; audit trails |
| **Data leakage via retrieval** | User sees chunks they're not authorized for | **ACL pre-filtering** synced with the source IdP; per-tenant isolation; never cache answers across permission sets; test with permission-matrix tests |
| **Sensitive information disclosure** | PII, secrets, or the system prompt leaked | PII redaction (ingest and output), secret scanning at ingest, don't put secrets in prompts |
| **Excessive agency** | Agent takes harmful actions | Tool allow-lists, human-in-the-loop for writes, scoped credentials, rate limits |
| **Hallucination / misinformation** | Ungrounded claims | Groundedness checks, citations, abstention, domain disclaimers |
| **Embedding inversion / vector leakage** | Text reconstructed from embeddings (Vec2Text, Morris et al. 2023) | Treat vectors as sensitive as the text; encrypt; restrict access |
| **Denial of wallet** | Abuse that drives up LLM cost | Rate limits, per-user budgets, agent step caps |
| **Supply chain** | Malicious models or packages | Pin and verify, scan model files (safetensors), SBOM |

## 2. Guardrail pipeline

```
INPUT  : authN/Z → rate limit → PII detect/redact → injection/jailbreak classifier
         → topic/scope policy → (block | safe-complete | proceed)
RETRIEVAL: ACL pre-filter → provenance/authority weighting → content-injection scan on chunks
OUTPUT : groundedness check (claim vs. sources; e.g. NLI, judge, or Bespoke-MiniCheck-style models)
         → citation validation → PII/secret scan → policy/compliance checks → format check
         → (deliver | regenerate | fallback "I can't answer that" | escalate to human)
```

DoorDash's two-tier guardrail: a cheap first check, then an LLM evaluation of groundedness, coherence, and compliance only when needed. They report ~90% fewer hallucinations and ~99% fewer severe compliance issues *(company-reported)*.

## 3. ACL-aware retrieval in detail
- Ingest ACLs per document (users, groups, roles), plus inheritance from folders and spaces.
- Resolve the requesting user's group memberships at query time (cache with a short TTL).
- Filter **inside the vector/BM25 query** (`acl_groups ∈ user_groups`) before top-K.
- Permission changes in the source must propagate (an SLA of minutes to hours); handle "deny" overrides.
- **Tests:** a matrix of users × documents, asserting no unauthorized chunk *ever* appears in the retrieved set (not just the answer).
- Response caches keyed by permission scope (or disabled for permissioned content).

## 4. Hallucination controls (ranked by ROI)
1. Better retrieval (most hallucinations are retrieval misses).
2. Strict grounding instructions + an abstention path.
3. Claim-level citation requirements + verification.
4. Groundedness classifier on outputs (block or regenerate).
5. Lower temperature; more faithful models (measured).
6. UX: show sources; confidence indicators; "verify in source" links.

## 5. Red-teaming
- Build an adversarial suite: jailbreaks, indirect injections in test documents, ACL probes, PII extraction, off-topic, harmful-advice prompts, multilingual attacks.
- Tools: garak, PyRIT, promptfoo red-team, Giskard.
- Run it in CI; track attack success rate over time.

## Expert checks
- Why is indirect prompt injection fundamentally harder than direct injection in RAG?
- How would you prove your system never retrieves unauthorized chunks?
- What's the exfiltration risk of rendering markdown images in answers?
