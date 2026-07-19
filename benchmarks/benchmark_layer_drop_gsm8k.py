# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GSM8K math-reasoning accuracy benchmark for layer drop.

Loads examples from the GSM8K test split, sends them to baseline and
layer-drop servers, extracts the final numerical answer, and reports
answer-level accuracy as well as the drop ratio.
"""

import argparse
import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests


def normalize_number(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = re.sub(r"\.$", "", s)
    return s


def extract_final_number(text: str) -> str | None:
    """Extract the final numerical answer from a GSM8K-style response."""
    if not text:
        return None
    if "####" in text:
        match = re.search(r"####\s*(-?[\d]+\.?\d*)", text)
        if match:
            return normalize_number(match.group(1))
    patterns = [
        r"answer is\s+(-?[\d]+\.?\d*)",
        r"answer:\s*(-?[\d]+\.?\d*)",
        r"final answer is\s+(-?[\d]+\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1])
    matches = re.findall(r"(-?[\d]+\.?\d*)", text)
    if matches:
        return normalize_number(matches[-1])
    return None


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer after '####'.\n\n"
        f"Question: {question}\nAnswer:"
    )


def load_gsm8k(num_prompts: int, seed: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library is required. Install with: uv pip install datasets"
        ) from e

    random.seed(seed)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = random.sample(range(len(ds)), min(num_prompts, len(ds)))
    results: list[dict[str, Any]] = []
    for i in indices:
        item = ds[i]
        question = item["question"].strip()
        gold = item["answer"].split("####")[-1].strip()
        results.append({
            "question": question,
            "prompt": build_prompt(question),
            "answer": normalize_number(gold),
        })
    return results


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
        }
    except Exception as e:
        return {"error": str(e)}


def collect_responses(
    items: list[dict[str, Any]],
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    label: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = [None] * len(items)
    threads: list[threading.Thread] = []

    def worker(idx: int, prompt: str) -> None:
        print(f"[{idx + 1}/{len(items)}] {label}")
        responses[idx] = send_completion(url, model, prompt, max_tokens,
                                         temperature)

    for i, item in enumerate(items):
        t = threading.Thread(target=worker, args=(i, item["prompt"]))
        threads.append(t)
        t.start()
        if len(threads) >= concurrency:
            for t in threads:
                t.join()
            threads = []

    for t in threads:
        t.join()
    return responses


def evaluate_item(
    item: dict[str, Any],
    base: dict[str, Any],
    ld: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "question": item["question"][:200],
        "gold_answer": item["answer"],
        "baseline": base,
        "layer_drop": ld,
    }
    if "error" in base or "error" in ld:
        result["error"] = True
        return result

    base_pred = extract_final_number(base["text"])
    ld_pred = extract_final_number(ld["text"])
    result["baseline_pred"] = base_pred
    result["layer_drop_pred"] = ld_pred
    result["baseline_correct"] = (base_pred == item["answer"])
    result["layer_drop_correct"] = (ld_pred == item["answer"])
    result["dropped"] = ld["finish_reason"] == "length" and ld["text"] == ""
    return result


def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in results if not r.get("error")]
    dropped = [r for r in valid if r.get("dropped")]
    answered = [r for r in valid if not r.get("dropped")]
    return {
        "num_prompts": len(results),
        "valid": len(valid),
        "dropped_count": len(dropped),
        "drop_ratio": len(dropped) / len(valid) if valid else 0.0,
        "answered_count": len(answered),
        "baseline_accuracy": float(
            np.mean([r["baseline_correct"] for r in answered])
            if answered else 0.0),
        "layer_drop_accuracy": float(
            np.mean([r["layer_drop_correct"] for r in answered])
            if answered else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["baseline", "layer_drop", "both"], default="both")
    parser.add_argument("--baseline-url",
                        default="http://localhost:8000/v1/completions")
    parser.add_argument("--layer-drop-url",
                        default="http://localhost:8001/v1/completions")
    parser.add_argument("--baseline-cache",
                        default="results/layer_drop_gsm8k_baseline_cache.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="results/layer_drop_gsm8k.json")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = load_gsm8k(args.num_prompts, args.seed)

    if args.mode == "baseline":
        baseline_responses = collect_responses(
            items, args.baseline_url, args.model, args.max_tokens,
            args.temperature, "baseline", args.concurrency)
        cache = {
            "config": {
                "url": args.baseline_url,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            "items": items,
            "responses": baseline_responses,
        }
        cache_path = Path(args.baseline_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"Saved baseline responses to {cache_path}")
        return

    if args.mode == "layer_drop":
        cache_path = Path(args.baseline_cache)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Baseline cache not found: {cache_path}. "
                "Run with --mode baseline first.")
        with cache_path.open("r") as f:
            cache = json.load(f)
        baseline_responses = cache["responses"]
        layer_drop_responses = collect_responses(
            items, args.layer_drop_url, args.model, args.max_tokens,
            args.temperature, "layer_drop", args.concurrency)
        results = [
            evaluate_item(it, b, l)
            for it, b, l in zip(items, baseline_responses,
                                layer_drop_responses)
        ]
    else:  # both
        results = []
        for i, item in enumerate(items):
            print(f"[{i + 1}/{len(items)}] comparing")
            base = send_completion(args.baseline_url, args.model, item["prompt"],
                                   args.max_tokens, args.temperature)
            ld = send_completion(args.layer_drop_url, args.model, item["prompt"],
                                 args.max_tokens, args.temperature)
            results.append(evaluate_item(item, base, ld))

    summary = compute_summary(results)
    output = {
        "config": {
            "mode": args.mode,
            "baseline_url": args.baseline_url,
            "layer_drop_url": args.layer_drop_url,
            "baseline_cache": args.baseline_cache,
            "model": args.model,
            "num_prompts": args.num_prompts,
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
    print(f"Wrote GSM8K benchmark results to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()