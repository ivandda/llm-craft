"""Offline coverage/accuracy scoring for generated recipe predictions.

Generation is the expensive (GPU) part and is saved to predictions.jsonl by
src/eval/run_sft_eval.py. This module scores those saved outputs on CPU, so any
number of models can be compared and re-scored for free, without re-running them.

Primary metric: coverage/accuracy against the recipe's known_outputs (normalized
string match), reported both for the top-1 sample and as any@K over all samples.
An optional semantic coverage (embedding cosine >= threshold) credits outputs that
are correct but not an exact string. Verbosity is tracked too, since the task
target is <=2 words.

Usage:
    python -m src.eval.score_coverage PREDICTIONS_A.jsonl PREDICTIONS_B.jsonl
    python -m src.eval.score_coverage gs://bucket/eval_outputs/run/predictions.jsonl
    python -m src.eval.score_coverage --labels base,concept_set a.jsonl b.jsonl
    python -m src.eval.score_coverage --semantic_threshold 0.7 a.jsonl b.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

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


def _max_cosine_to_known(sample_vec: np.ndarray, known_matrix: np.ndarray) -> float:
    sample_norm = np.linalg.norm(sample_vec)
    if sample_norm == 0 or known_matrix.size == 0:
        return 0.0
    known_norms = np.linalg.norm(known_matrix, axis=1)
    known_norms[known_norms == 0] = 1.0
    cosines = (known_matrix @ sample_vec) / (known_norms * sample_norm)
    return float(cosines.max())


def semantic_coverage(records: list[dict[str, Any]], embedder: Any, threshold: float) -> dict[str, float]:
    """Coverage where a sample counts if its cosine to any known output >= threshold."""
    n = 0
    top1 = 0
    anyk = 0
    for record in records:
        samples = record.get("sampled_outputs") or [record.get("prediction", "")]
        known_outputs = record.get("known_outputs") or []
        if not known_outputs:
            continue
        n += 1
        known_matrix = np.asarray(embedder.encode(known_outputs), dtype=float)
        sample_matrix = np.asarray(embedder.encode(samples), dtype=float)
        sims = [_max_cosine_to_known(sample_matrix[i], known_matrix) for i in range(len(samples))]
        if sims and sims[0] >= threshold:
            top1 += 1
        if any(sim >= threshold for sim in sims):
            anyk += 1
    if n == 0:
        return {"top1_semantic_match": 0.0, "anyk_semantic_match": 0.0, "semantic_threshold": threshold}
    return {
        "top1_semantic_match": top1 / n,
        "anyk_semantic_match": anyk / n,
        "semantic_threshold": threshold,
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
    has_semantic = any("top1_semantic_match" in scores for _, scores in labeled_scores)
    columns: list[tuple[str, Any]] = [
        ("model", lambda label, s: label),
        ("n", lambda label, s: str(int(s["n"]))),
        ("top1_known", lambda label, s: f"{s['top1_known_match']:.3f}"),
        ("any@k_known", lambda label, s: f"{s['anyk_known_match']:.3f}"),
        ("top1_canon", lambda label, s: f"{s['top1_canonical_match']:.3f}"),
        ("any@k_canon", lambda label, s: f"{s['anyk_canonical_match']:.3f}"),
    ]
    if has_semantic:
        columns += [
            ("top1_sem", lambda label, s: f"{s.get('top1_semantic_match', float('nan')):.3f}"),
            ("any@k_sem", lambda label, s: f"{s.get('anyk_semantic_match', float('nan')):.3f}"),
        ]
    columns += [
        ("empty", lambda label, s: f"{s['empty_top1_rate']:.3f}"),
        ("words", lambda label, s: f"{s['mean_top1_words']:.2f}"),
        ("<=2w", lambda label, s: f"{s['frac_top1_le2_words']:.3f}"),
    ]

    rows = [[title for title, _ in columns]]
    for label, scores in labeled_scores:
        rows.append([getter(label, scores) for _, getter in columns])
    widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]

    lines = []
    for row_index, row in enumerate(rows):
        cells = [
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        ]
        line = "  ".join(cells)
        lines.append(line)
        if row_index == 0:
            lines.append("-" * len(line))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", help="One or more predictions.jsonl paths (local or gs://).")
    parser.add_argument("--labels", default=None, help="Comma-separated labels, one per predictions file.")
    parser.add_argument("--json", action="store_true", help="Also print the raw scores as JSON.")
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=None,
        help="If set, also compute embedding-based semantic coverage (cosine >= threshold).",
    )
    parser.add_argument("--embedding_backend", default="sentence_embeddings")
    parser.add_argument("--embedding_model_path", default=None)
    parser.add_argument("--sentence_embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    labels = args.labels.split(",") if args.labels else [_default_label(path) for path in args.predictions]
    if len(labels) != len(args.predictions):
        raise ValueError("Number of --labels must match number of predictions files.")

    embedder = None
    if args.semantic_threshold is not None:
        from src.eval.embeddings import build_text_embedder

        embedder = build_text_embedder(
            args.embedding_backend,
            word_vector_path=args.embedding_model_path,
            sentence_transformer_model=args.sentence_embedding_model,
            device=args.embedding_device,
        )

    labeled_scores: list[tuple[str, dict[str, float]]] = []
    for label, path in zip(labels, args.predictions):
        records = load_prediction_records(path)
        scores = score_records(records)
        if embedder is not None:
            scores.update(semantic_coverage(records, embedder, args.semantic_threshold))
        labeled_scores.append((label, scores))

    print(format_comparison(labeled_scores))
    if args.json:
        print()
        print(json.dumps({label: scores for label, scores in labeled_scores}, indent=2))


if __name__ == "__main__":
    main()
