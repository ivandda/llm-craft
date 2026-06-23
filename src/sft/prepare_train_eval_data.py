import argparse
import hashlib
import heapq
import json
import os
from dataclasses import dataclass


@dataclass(order=True)
class SampleCandidate:
    score: int
    index: int
    line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic conversational train/dev samples for SFT runs."
    )
    parser.add_argument(
        "--train-input",
        default="datasets/processed/sft_clean_train.jsonl",
        help="Path to the full conversational train JSONL.",
    )
    parser.add_argument(
        "--eval-input",
        default="datasets/processed/sft_clean_dev.jsonl",
        help="Path to the full conversational dev JSONL.",
    )
    parser.add_argument(
        "--train-output",
        default="artifacts/data/sft_clean_train_sample_8000.jsonl",
        help="Path where the train sample will be written.",
    )
    parser.add_argument(
        "--eval-output",
        default="artifacts/data/sft_clean_dev_sample_2000.jsonl",
        help="Path where the eval sample will be written.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=8000,
        help="Number of train examples to keep.",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=2000,
        help="Number of eval examples to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to derive deterministic hash scores.",
    )
    return parser.parse_args()


def stable_score(recipe_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{recipe_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def recipe_id_from_record(record: dict, fallback_index: int) -> str:
    metadata = record.get("metadata", {})
    recipe_id = metadata.get("recipe_id")
    if recipe_id:
        return recipe_id
    return f"line-{fallback_index}"


def sample_jsonl(input_path: str, output_path: str, sample_size: int, seed: int) -> dict[str, int | str]:
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    selected: list[tuple[int, int, str]] = []
    total_rows = 0

    with open(input_path, "r", encoding="utf-8") as infile:
        for index, raw_line in enumerate(infile):
            if not raw_line.strip():
                continue

            total_rows += 1
            record = json.loads(raw_line)
            recipe_id = recipe_id_from_record(record, index)
            score = stable_score(recipe_id, seed)

            candidate = (-score, index, raw_line)
            if len(selected) < sample_size:
                heapq.heappush(selected, candidate)
                continue

            current_worst_score = -selected[0][0]
            if score < current_worst_score:
                heapq.heapreplace(selected, candidate)

    sampled_rows = sorted(
        (SampleCandidate(score=-score, index=index, line=line) for score, index, line in selected),
        key=lambda item: item.index,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outfile:
        for row in sampled_rows:
            outfile.write(row.line)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "sample_size": len(sampled_rows),
        "total_rows_seen": total_rows,
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    train_summary = sample_jsonl(
        input_path=args.train_input,
        output_path=args.train_output,
        sample_size=args.train_sample_size,
        seed=args.seed,
    )
    eval_summary = sample_jsonl(
        input_path=args.eval_input,
        output_path=args.eval_output,
        sample_size=args.eval_sample_size,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "train": train_summary,
                "eval": eval_summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
