"""Offline coverage/accuracy scoring for generated recipe predictions.

Generation is the expensive (GPU) part and is saved to predictions.jsonl by
src/eval/run_sft_eval.py. This module scores those saved outputs on CPU, so any
number of models can be compared and re-scored for free, without re-running them.

Primary metric: coverage/accuracy against the recipe's known_outputs (normalized
string match), reported both for the top-1 sample and as any@K over all samples.
Verbosity is tracked too, since the task target is <=2 words.

Usage:
    python -m src.eval.score_coverage PREDICTIONS_A.jsonl PREDICTIONS_B.jsonl
    python -m src.eval.score_coverage gs://bucket/eval_outputs/run/predictions.jsonl
    python -m src.eval.score_coverage --labels base,concept_set a.jsonl b.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from src.eval.metrics import evaluate_prediction, normalize_answer


def _word_count(text: str | None) -> int:
    return len(normalize_answer(text).split())


def score_records(records: list[dict[str, Any]]) -> dict[str, float]:
    n = 0
    top1_known = top1_canon = anyk_known = anyk_canon = empty_top1 = 0
    total_words = 0
    le2_words = 0
    for record in records:
        samples = record.get("sampled_outputs") or []
        if not samples:
            samples = [record.get("prediction", "")]
        known_outputs = record.get("known_outputs") or []
        canonical = record.get("canonical_output") or (known_outputs[0] if known_outputs else "")

        n += 1
        top1_eval = evaluate_prediction(samples[0], canonical, known_outputs)
        top1_known += int(top1_eval.known_output_match)
        top1_canon += int(top1_eval.exact_canonical_match)
        empty_top1 += int(top1_eval.is_empty_prediction)

        words = _word_count(samples[0])
        total_words += words
        le2_words += int(0 < words <= 2)

        sample_evals = [evaluate_prediction(sample, canonical, known_outputs) for sample in samples]
        anyk_known += int(any(sample_eval.known_output_match for sample_eval in sample_evals))
        anyk_canon += int(any(sample_eval.exact_canonical_match for sample_eval in sample_evals))

    if n == 0:
        return {
            "n": 0,
            "top1_known_match": 0.0,
            "top1_canonical_match": 0.0,
            "anyk_known_match": 0.0,
            "anyk_canonical_match": 0.0,
            "empty_top1_rate": 0.0,
            "mean_top1_words": 0.0,
            "frac_top1_le2_words": 0.0,
        }
    return {
        "n": n,
        "top1_known_match": top1_known / n,
        "top1_canonical_match": top1_canon / n,
        "anyk_known_match": anyk_known / n,
        "anyk_canonical_match": anyk_canon / n,
        "empty_top1_rate": empty_top1 / n,
        "mean_top1_words": total_words / n,
        "frac_top1_le2_words": le2_words / n,
    }


def load_prediction_records(path: str) -> list[dict[str, Any]]:
    if path.startswith("gs://"):
        completed = subprocess.run(
            ["gsutil", "cat", path], capture_output=True, text=True, check=True
        )
        lines = completed.stdout.splitlines()
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _default_label(path: str) -> str:
    parent = Path(path).parent.name
    return parent or Path(path).stem


def format_comparison(labeled_scores: list[tuple[str, dict[str, float]]]) -> str:
    columns = [
        ("model", 34, "{:<34}"),
        ("n", 6, "{:>6}"),
        ("top1_known", 11, "{:>11.3f}"),
        ("any@k_known", 12, "{:>12.3f}"),
        ("top1_canon", 11, "{:>11.3f}"),
        ("any@k_canon", 12, "{:>12.3f}"),
        ("empty", 7, "{:>7.3f}"),
        ("words", 7, "{:>7.2f}"),
        ("<=2w", 7, "{:>7.3f}"),
    ]
    keys = [
        None,
        "n",
        "top1_known_match",
        "anyk_known_match",
        "top1_canonical_match",
        "anyk_canonical_match",
        "empty_top1_rate",
        "mean_top1_words",
        "frac_top1_le2_words",
    ]
    header = " ".join(f"{name:<{width}}" if fmt.startswith("{:<") else f"{name:>{width}}"
                       for (name, width, fmt), _ in zip(columns, keys))
    lines = [header, "-" * len(header)]
    for label, scores in labeled_scores:
        cells = []
        for (name, width, fmt), key in zip(columns, keys):
            if key is None:
                cells.append(fmt.format(label[:width]))
            elif key == "n":
                cells.append(fmt.format(int(scores[key])))
            else:
                cells.append(fmt.format(scores[key]))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", help="One or more predictions.jsonl paths (local or gs://).")
    parser.add_argument("--labels", default=None, help="Comma-separated labels, one per predictions file.")
    parser.add_argument("--json", action="store_true", help="Also print the raw scores as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    labels = args.labels.split(",") if args.labels else [_default_label(path) for path in args.predictions]
    if len(labels) != len(args.predictions):
        raise ValueError("Number of --labels must match number of predictions files.")

    labeled_scores: list[tuple[str, dict[str, float]]] = []
    for label, path in zip(labels, args.predictions):
        records = load_prediction_records(path)
        labeled_scores.append((label, score_records(records)))

    print(format_comparison(labeled_scores))
    if args.json:
        print()
        print(json.dumps({label: scores for label, scores in labeled_scores}, indent=2))


if __name__ == "__main__":
    main()
