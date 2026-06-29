from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.metrics import evaluate_prediction


DEFAULT_EVAL_FILE = "datasets/processed/eval_dev_all.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT predictions on JSONL data.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def build_output_record(eval_record: dict[str, Any], prediction: str | None) -> dict[str, Any]:
    known_outputs = list(eval_record.get("known_outputs", []))
    evaluation = evaluate_prediction(
        prediction,
        str(eval_record.get("canonical_output", "")),
        known_outputs,
    )

    return {
        "pair_id": eval_record.get("pair_id"),
        "input_a": eval_record.get("input_a"),
        "input_b": eval_record.get("input_b"),
        "prediction": prediction,
        "canonical_output": eval_record.get("canonical_output"),
        "known_outputs": known_outputs,
        "exact_canonical_match": evaluation.exact_canonical_match,
        "known_output_match": evaluation.known_output_match,
        "is_empty_prediction": evaluation.is_empty_prediction,
    }


def summarize_output_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_examples": 0,
            "canonical_accuracy": 0.0,
            "known_output_accuracy": 0.0,
            "empty_predictions": 0,
        }

    return {
        "num_examples": len(records),
        "canonical_accuracy": sum(
            1 for record in records if record["exact_canonical_match"]
        )
        / len(records),
        "known_output_accuracy": sum(
            1 for record in records if record["known_output_match"]
        )
        / len(records),
        "empty_predictions": sum(
            1 for record in records if record["is_empty_prediction"]
        ),
    }


def main() -> None:
    args = parse_args()
    eval_records = list(iter_jsonl(Path(args.eval_file)))
    predictions = read_predictions(args.predictions_file, len(eval_records))
    output_records = [
        build_output_record(record, prediction)
        for record, prediction in zip(eval_records, predictions, strict=True)
    ]

    if args.output_file:
        write_jsonl(Path(args.output_file), output_records)

    print(json.dumps(summarize_output_records(output_records), indent=2))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_predictions(path: str | None, expected_count: int) -> list[str | None]:
    if path is None:
        return [None] * expected_count

    records = list(iter_jsonl(Path(path)))
    return [record.get("prediction") for record in records]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
