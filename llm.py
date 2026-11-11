import argparse
import json
import os
from pathlib import Path

import numpy as np

from baselines.llm.plain import complete, messages
from common.data import load_data, save_predictions

API_KEY = "YOUR_API_KEY"  # Replace here, or set LLM_API_KEY in your environment.
METHODS = ("deepseek", "baichuan", "curated-llm", "tabgen-icl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, default="deepseek")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "YOUR_MODEL_ID"))
    parser.add_argument("--context", type=int, help="Context count (20 for plain LLMs; "
                        "1 per unique target value on the first TabGen-ICL prompt).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5, help="Generated samples per "
                        "query and target for TabGen-ICL; predictions use their mean.")
    parser.add_argument("--max-prompt-iters", type=int, default=10)
    parser.add_argument("--curation-label", default="label")
    parser.add_argument("--curation-trees", type=int, default=600)
    parser.add_argument("--curation-x-threshold", type=float, default=0.2)
    parser.add_argument("--curation-y-threshold", type=float, default=0.2)
    parser.add_argument("--retries-limit", type=int, default=3,
                        help="Consecutive curation rounds with no accepted rows.")
    parser.add_argument("--concurrency", type=int, help="Concurrent requests: "
                        "100 for Curated LLM, 8 for TabGen-ICL.")
    parser.add_argument("--temperature", type=float, help="0.7 for plain/Curated LLM; "
                        "1.0 for TabGen-ICL.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Save the first prompt without calling the API.")
    args = parser.parse_args()
    args.context = args.context if args.context is not None else (1 if args.method == "tabgen-icl" else 20)
    args.temperature = args.temperature if args.temperature is not None else (1.0 if args.method == "tabgen-icl" else 0.7)
    args.concurrency = args.concurrency if args.concurrency is not None else (100 if args.method == "curated-llm" else 8)
    if min(args.context, args.batch_size, args.samples, args.max_tokens,
           args.max_prompt_iters, args.retries_limit, args.concurrency) < 1:
        parser.error("Counts and limits must be positive.")
    if args.curation_trees < 2:
        parser.error("curation-trees must be >= 2.")
    if not 0 <= args.curation_x_threshold <= 1 or not 0 <= args.curation_y_threshold < 0.5:
        parser.error("curation-x-threshold must be in [0, 1] and curation-y-threshold in [0, 0.5).")
    if not np.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        parser.error("temperature must be in [0, 2].")
    key = os.getenv("LLM_API_KEY", API_KEY)
    if not args.dry_run and (not key or key == "YOUR_API_KEY" or args.model == "YOUR_MODEL_ID"):
        parser.error("Set LLM_API_KEY (or API_KEY in llm.py) and specify --model.")
    x_train, y_train, query = load_data(args.data)
    rng = np.random.default_rng(args.seed)
    output = args.output or Path("outputs") / args.method
    protocol = None
    if args.method == "curated-llm":
        from baselines.llm import curated as protocol
        protocol.training_table(x_train, y_train, args.curation_label)
    elif args.method == "tabgen-icl":
        from baselines.llm import tabgen_icl as protocol
    if args.dry_run:
        rows = np.arange(min(args.batch_size, len(query)))
        if args.method == "curated-llm":
            prompt = protocol.preview(x_train, y_train, query, args.curation_label)
        elif args.method == "tabgen-icl":
            prompt = protocol.preview(x_train, y_train, query, args.seed, args.context)
        else:
            context = rng.choice(len(x_train), min(args.context, len(x_train)), replace=False)
            prompt = messages(x_train, y_train, query, rows, context)
        request = {"model": args.model, "temperature": args.temperature,
                   "max_tokens": args.max_tokens, "messages": prompt}
        output.mkdir(parents=True, exist_ok=True)
        path = output / "request.json"
        path.write_text(json.dumps(request, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved prompt to {path}")
        return
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=args.base_url, timeout=120, max_retries=2)
    details = {}
    if args.method == "curated-llm":
        prediction, details = protocol.predict(
            client, args.model, x_train, y_train, query, label=args.curation_label,
            temperature=args.temperature, max_tokens=args.max_tokens,
            nest=args.curation_trees, retries_limit=args.retries_limit,
            x_threshold=args.curation_x_threshold, y_threshold=args.curation_y_threshold,
            concurrency=args.concurrency,
        )
    elif args.method == "tabgen-icl":
        prediction, details = protocol.predict(
            client, args.model, x_train, y_train, query, seed=args.seed,
            ic_samples=args.context, n_samples=args.samples,
            max_prompt_iters=args.max_prompt_iters, temperature=args.temperature,
            max_tokens=args.max_tokens, concurrency=args.concurrency,
        )
    else:
        predictions = []
        for start in range(0, len(query), args.batch_size):
            rows = np.arange(start, min(start + args.batch_size, len(query)))
            context = rng.choice(len(x_train), min(args.context, len(x_train)), replace=False)
            prompt = messages(x_train, y_train, query, rows, context)
            predictions.append(complete(client, args.model, prompt, rows, list(y_train.columns),
                                        args.temperature, args.max_tokens))
            print(f"{args.method}: {rows[-1] + 1}/{len(query)}")
        prediction = np.concatenate(predictions)
    save_predictions(prediction, y_train.columns, args.data, output,
                     method=args.method, model=args.model, base_url=args.base_url,
                     seed=args.seed, context=args.context, **details)
    if details:
        (output / "generation_log.json").write_text(
            json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
