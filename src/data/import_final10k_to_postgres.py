from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from src.data.db import connect, repo_root


SPLITS = ("train", "dev", "test")
DEFAULT_DATASET_NAME = "final-10k"


def stable_id(parts: list[str]) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_concept(value: str) -> str:
    return " ".join(value.strip().lower().split())


def pair_values(input_a: str, input_b: str) -> tuple[str, str]:
    return tuple(sorted([normalize_concept(input_a), normalize_concept(input_b)]))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import datasets/final-10k into Postgres.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-dir",
        default=str(repo_root() / "datasets" / "final-10k"),
        help="Path containing train/dev/test/rejected jsonl files.",
    )
    parser.add_argument(
        "--replace-dataset",
        metavar="DATASET_NAME",
        default=None,
        help="Delete and re-import dataset rows for this dataset name.",
    )
    return parser.parse_args()


def read_manifest(dataset_dir: Path, name: str) -> dict[str, Any] | None:
    path = dataset_dir / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def insert_dataset_import(connection, dataset_name: str, dataset_dir: Path) -> None:
    metadata = {
        "imported_by": "src.data.import_final10k_to_postgres",
        "source_files": ["train.jsonl", "dev.jsonl", "test.jsonl", "rejected.jsonl"],
    }
    connection.execute(
        """
        INSERT INTO dataset_imports (dataset_name, source_dir, raw_metadata)
        VALUES (%s, %s, %s)
        ON CONFLICT (dataset_name) DO UPDATE SET
          source_dir = EXCLUDED.source_dir,
          imported_at = now(),
          raw_metadata = EXCLUDED.raw_metadata
        """,
        (dataset_name, str(dataset_dir), Jsonb(metadata)),
    )


def import_split(connection, dataset_name: str, dataset_dir: Path, split: str) -> int:
    count = 0
    path = dataset_dir / f"{split}.jsonl"
    for line_number, record in iter_jsonl(path):
        input_a, input_b = pair_values(record.get("input_a", ""), record.get("input_b", ""))
        pair_id = stable_id([dataset_name, input_a, input_b])
        connection.execute(
            """
            INSERT INTO recipe_pairs (
              pair_id, dataset_name, split, input_a, input_b, raw_record
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (pair_id, dataset_name, split, input_a, input_b, Jsonb(record)),
        )
        candidates = record.get("candidate_outputs", [])
        for index, candidate in enumerate(candidates, start=1):
            rank = int(candidate.get("rank") or index)
            output = normalize_concept(candidate.get("output", ""))
            candidate_id = stable_id([pair_id, str(rank), output])
            connection.execute(
                """
                INSERT INTO recipe_candidates (
                  candidate_id, pair_id, output, source, rationale, rank, raw_candidate
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    candidate_id,
                    pair_id,
                    output,
                    candidate.get("source", "teacher"),
                    candidate.get("rationale"),
                    rank,
                    Jsonb(candidate),
                ),
            )
        count += 1
    return count


def import_rejections(connection, dataset_name: str, dataset_dir: Path) -> int:
    count = 0
    path = dataset_dir / "rejected.jsonl"
    for line_number, record in iter_jsonl(path):
        input_a, input_b = pair_values(record.get("input_a", ""), record.get("input_b", ""))
        split = normalize_concept(record.get("split", "unknown")) or "unknown"
        rejection_id = stable_id([dataset_name, split, str(line_number), input_a, input_b])
        connection.execute(
            """
            INSERT INTO dataset_rejections (
              rejection_id, dataset_name, split, input_a, input_b, outputs,
              reject_reason, detail, raw_record
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                rejection_id,
                dataset_name,
                split,
                input_a,
                input_b,
                Jsonb(record.get("outputs", [])),
                record.get("reject_reason", ""),
                record.get("detail"),
                Jsonb(record),
            ),
        )
        count += 1
    return count


def import_manifests(connection, dataset_name: str, dataset_dir: Path) -> None:
    for manifest_name in ("batch_export_manifest.json", "batch_import_manifest.json"):
        manifest = read_manifest(dataset_dir, manifest_name)
        if manifest is None:
            continue
        manifest_id = stable_id([dataset_name, manifest_name])
        connection.execute(
            """
            INSERT INTO dataset_manifests (
              manifest_id, dataset_name, manifest_name, raw_manifest
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (dataset_name, manifest_name) DO UPDATE SET
              raw_manifest = EXCLUDED.raw_manifest,
              created_at = now()
            """,
            (manifest_id, dataset_name, manifest_name, Jsonb(manifest)),
        )


def import_batch_index_manifest(connection, dataset_name: str, dataset_dir: Path) -> None:
    path = dataset_dir / "batch_index.jsonl"
    if not path.exists():
        return
    count = sum(1 for _line_number, _record in iter_jsonl(path))
    manifest_id = stable_id([dataset_name, "batch_index.jsonl"])
    payload = {
        "file": "batch_index.jsonl",
        "record_count": count,
        "preserved_on_disk": True,
    }
    connection.execute(
        """
        INSERT INTO dataset_manifests (
          manifest_id, dataset_name, manifest_name, raw_manifest
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (dataset_name, manifest_name) DO UPDATE SET
          raw_manifest = EXCLUDED.raw_manifest,
          created_at = now()
        """,
        (manifest_id, dataset_name, "batch_index.jsonl", Jsonb(payload)),
    )


def delete_dataset_rows(connection, dataset_name: str) -> None:
    connection.execute("DELETE FROM dataset_imports WHERE dataset_name = %s", (dataset_name,))


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name
    dataset_dir = Path(args.dataset_dir)
    if args.replace_dataset:
        dataset_name = args.replace_dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    with connect() as connection:
        if args.replace_dataset:
            delete_dataset_rows(connection, args.replace_dataset)
        insert_dataset_import(connection, dataset_name, dataset_dir)
        split_counts = {
            split: import_split(connection, dataset_name, dataset_dir, split)
            for split in SPLITS
        }
        rejected_count = import_rejections(connection, dataset_name, dataset_dir)
        import_manifests(connection, dataset_name, dataset_dir)
        import_batch_index_manifest(connection, dataset_name, dataset_dir)
        connection.execute(
            """
            UPDATE dataset_imports
            SET train_count = %s,
                dev_count = %s,
                test_count = %s,
                rejected_count = %s,
                imported_at = %s
            WHERE dataset_name = %s
            """,
            (
                split_counts["train"],
                split_counts["dev"],
                split_counts["test"],
                rejected_count,
                datetime.now(UTC),
                dataset_name,
            ),
        )
    print(
        json.dumps(
            {
                "dataset_name": dataset_name,
                **split_counts,
                "rejected": rejected_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
