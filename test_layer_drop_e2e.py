import argparse
import threading
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-short", type=int, default=9)
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()

    long_prompt = (
        "In the year 2147, humanity discovered a way to travel faster than light. "
        "The first expedition to the Andromeda galaxy was planned for over a decade "
        "and involved thousands of scientists from every nation on Earth. The mission "
        "was called Project Starlight, and its goal was not only to explore but also "
        "to find a new home for mankind."
    )

    results = []

    def send(prompt, idx):
        start = time.time()
        r = requests.post(
            args.url,
            json={
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
            },
        )
        latency = time.time() - start
        data = r.json()
        text = data["choices"][0]["text"].strip()[:60]
        results.append((idx, latency, text))

    threads = []
    for i in range(args.num_short):
        t = threading.Thread(target=send, args=("Hi", i))
        threads.append(t)
    t = threading.Thread(target=send, args=(long_prompt, args.num_short))
    threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.time() - start

    print(f"\nTotal time: {total:.2f}s")
    for idx, latency, text in sorted(results):
        print(f"[{idx}] {latency:.2f}s: {text}")


if __name__ == "__main__":
    main()
