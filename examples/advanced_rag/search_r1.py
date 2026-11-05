"""RL-trained search agents: Search-R1 (Jin et al., 2025), R1-Searcher, ReSearch,
ZeroSearch, DeepResearcher. The policy LLM learns WHEN and WHAT to search purely from an
outcome reward. Rollout format: <think> ... <search>query</search> <information>retrieved
</information> ... <answer>...</answer>. Key ingredients, all shown below:
  1. multi-turn rollouts where the environment (retriever) inserts <information>;
  2. a rule-based outcome reward (exact match), optionally plus a format reward;
  3. retrieved-token loss masking: the policy is not trained to predict <information>;
  4. a group-relative advantage (GRPO) over G rollouts of the same question."""

import re
import string
from collections.abc import Callable

import numpy as np
from rag_kit import Retriever, show

SYSTEM = ("Answer the question. Reason inside <think></think>. If you need knowledge, call "
          "the search engine with <search>query</search>; results come back inside "
          "<information></information>. Give the final answer inside <answer></answer>.")


def rollout(policy: Callable[[str], str], question: str, retriever: Retriever,
            max_turns: int = 4) -> dict:
    """``policy(prompt)`` continues the text and stops after </search> or </answer>."""
    text = f"{SYSTEM}\nQuestion: {question}\n"
    spans = []                                      # (start, end) of environment tokens
    for _ in range(max_turns):
        step = policy(text)
        text += step
        if m := re.search(r"<search>(.*?)</search>\s*$", step, re.S):
            docs = retriever.search(m.group(1).strip(), k=2)
            info = "\n<information>" + " ".join(d.text for d in docs) + "</information>\n"
            spans.append((len(text), len(text) + len(info)))
            text += info
        elif "</answer>" in step:
            break
    answer = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    return {"text": text, "answer": answer[-1].strip() if answer else "", "env_spans": spans}


def normalize(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(w for w in s.split() if w not in {"a", "an", "the"})


def reward(sample: dict, gold: list[str], format_bonus: float = 0.1) -> float:
    em = float(any(normalize(sample["answer"]) == normalize(g) for g in gold))
    generated = sample["text"].split("Question:", 1)[-1]
    well_formed = generated.count("<answer>") == 1 and "<think>" in generated
    return em + (format_bonus if well_formed else 0.0)


def loss_mask(sample: dict) -> np.ndarray:
    """1 for policy-generated characters (tokens in a real tokenizer), 0 for retrieved."""
    mask = np.ones(len(sample["text"]))
    for start, end in sample["env_spans"]:
        mask[start:end] = 0
    return mask


def grpo_advantages(rewards: list[float], eps: float = 1e-6) -> np.ndarray:
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + eps)


def grpo_loss(logp_new: np.ndarray, logp_old: np.ndarray, adv: float, mask: np.ndarray,
              clip: float = 0.2) -> float:
    """PPO-style clipped objective per sequence, averaged over UNMASKED tokens only.
    (Add a KL penalty to a reference model in practice; frameworks: veRL, TRL, OpenRLHF.)"""
    ratio = np.exp(logp_new - logp_old)
    obj = np.minimum(ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv)
    return float(-(obj * mask).sum() / max(mask.sum(), 1))


def scripted_policy(seed: int) -> Callable[[str], str]:
    """Stand-in for the LLM being trained: some rollouts search, some guess."""
    rng = np.random.default_rng(seed)

    def policy(text: str) -> str:
        searched = "<information>" in text.split("Question:", 1)[1]
        if not searched and rng.random() < 0.7:
            return "<think>I should look this up.</think><search>Lumen ST-2 factory</search>"
        city = "Gothenburg" if "Gothenburg" in text else "Lisbon"
        return f"<think>The evidence points to {city}.</think><answer>{city}</answer>"

    return policy


def demo() -> None:
    q, gold, r = "Where is the Lumen ST-2 built?", ["Gothenburg"], Retriever()
    group = [rollout(scripted_policy(s), q, r) for s in range(6)]     # G = 6 rollouts
    rewards = [reward(g, gold) for g in group]
    adv = grpo_advantages(rewards)
    losses = []
    for g, a in zip(group, adv, strict=True):
        mask = loss_mask(g)
        logp = np.full(len(g["text"]), -1.0)                          # placeholders
        losses.append(grpo_loss(logp, logp, float(a), mask))
    show("Search-R1 group", [{"answer": g["answer"], "searched": bool(g["env_spans"]),
                              "reward": rw, "advantage": round(float(a), 2)}
                             for g, rw, a in zip(group, rewards, adv, strict=True)])
    show("Mean GRPO loss (placeholder log-probs)", round(float(np.mean(losses)), 4))


if __name__ == "__main__":
    demo()
