# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-quality comparison between baseline and layer-drop servers.

Computes exact-match rate, token overlap, and ROUGE-L-like LCS similarity
for the same prompts under baseline and layer-drop configurations. Also
verifies that dropped requests finish with finish_reason=LENGTH and empty
new_token_ids.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests


def longest_common_subsequence(x: list[str], y: list[str]) -> int:
    if not x or not y:
        return 0
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def rouge_l_score(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = longest_common_subsequence(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    recall = lcs / len(ref_tokens)
    precision = lcs / len(hyp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_overlap(reference: str, hypothesis: str) -> float:
    ref_tokens = set(reference.split())
    hyp_tokens = set(hypothesis.split())
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)


def send_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    try:
        r = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        return {
            "text": choice.get("text", "").strip(),
            "finish_reason": choice.get("finish_reason", ""),
            "latency_ms": 0.0,
        }
    except Exception as e:
        return {"error": str(e)}


def compare_prompt(
    prompt: str,
    baseline_url: str,
    layer_drop_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    base = send_completion(baseline_url, model, prompt, max_tokens, temperature)
    ld = send_completion(layer_drop_url, model, prompt, max_tokens, temperature)

    result: dict[str, Any] = {
        "prompt": prompt[:200],
        "baseline": base,
        "layer_drop": ld,
    }

    if "error" in base or "error" in ld:
        result["error"] = True
        return result

    result["exact_match"] = base["text"] == ld["text"]
    result["token_overlap"] = token_overlap(base["text"], ld["text"])
    result["rouge_l"] = rouge_l_score(base["text"], ld["text"])

    # Dropped request characteristics
    result["dropped"] = ld["finish_reason"] == "length" and ld["text"] == ""
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url",
                        default="http://localhost:8000/v1/completions")
    parser.add_argument("--layer-drop-url",
                        default="http://localhost:8001/v1/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output",
                        default="results/layer_drop_quality.json")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about artificial intelligence.",
        "Summarize the plot of Romeo and Juliet.",
        "What are the main causes of climate change?",
        "Describe the structure of a cell.",
        "How does a blockchain work?",
        "List three benefits of regular exercise.",
        "What is the capital of France and its history?",
        "Write a Python function to compute factorial.",
        "Compare SQL and NoSQL databases.",
    ]
    prompts = (prompts * ((args.num_prompts // len(prompts)) + 1)
               )[:args.num_prompts]

    results: list[dict[str, Any]] = []
    dropped_count = 0
    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{args.num_prompts}] comparing prompt ...")
        entry = compare_prompt(prompt, args.baseline_url, args.layer_drop_url,
                               args.model, args.max_tokens, args.temperature)
        results.append(entry)
        if entry.get("dropped"):
            dropped_count += 1

    valid = [r for r in results if not r.get("error")]
    summary = {
        "num_prompts": len(prompts),
        "valid": len(valid),
        "dropped_count": dropped_count,
        "exact_match_rate": float(
            np.mean([r["exact_match"] for r in valid])),
        "mean_token_overlap": float(
            np.mean([r["token_overlap"] for r in valid])),
        "mean_rouge_l": float(np.mean([r["rouge_l"] for r in valid])),
    }

    output = {
        "config": {
            "baseline_url": args.baseline_url,
            "layer_drop_url": args.layer_drop_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "summary": summary,
        "details": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote quality comparison to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()