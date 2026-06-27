import argparse
import os
import zipfile
from pathlib import Path


DEFAULT_OUTPUT_PATH = "artifacts/colab/llm-craft-sft-colab.zip"
DEFAULT_TRAIN_FILE = "artifacts/data/recipes_train_sample_8000.jsonl"
DEFAULT_EVAL_FILE = "artifacts/data/recipes_dev_sample_2000.jsonl"
DEFAULT_STRUCTURED_EVAL_FILE = "datasets/processed/eval_dev_all.jsonl"

EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small Colab-ready zip with SFT code, config, docs, and sampled data."
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Where the Colab zip will be written.",
    )
    parser.add_argument(
        "--train-file",
        default=DEFAULT_TRAIN_FILE,
        help="Train JSONL sample to include in the zip.",
    )
    parser.add_argument(
        "--eval-file",
        default=DEFAULT_EVAL_FILE,
        help="Eval JSONL sample to include in the zip.",
    )
    parser.add_argument(
        "--eval-set-file",
        default=DEFAULT_STRUCTURED_EVAL_FILE,
        help="Structured evaluation JSONL to include for batch model scoring.",
    )
    parser.add_argument(
        "--zip-root",
        default="llm-craft-colab",
        help="Top-level directory name inside the zip.",
    )
    parser.add_argument(
        "--include-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include docs/ in the zip.",
    )
    return parser.parse_args()


def should_include(path: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    return path.suffix not in EXCLUDED_SUFFIXES


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and should_include(path))


def add_file(zip_handle: zipfile.ZipFile, source_path: Path, archive_path: Path) -> None:
    zip_handle.write(source_path, archive_path.as_posix())


def add_path(zip_handle: zipfile.ZipFile, source_path: Path, archive_root: Path) -> None:
    if source_path.is_file():
        add_file(zip_handle, source_path, archive_root / source_path.name)
        return

    for file_path in iter_files(source_path):
        add_file(zip_handle, file_path, archive_root / file_path.relative_to(source_path))


def require_existing(paths: list[Path]) -> None:
    missing = [path.as_posix() for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Cannot build Colab zip because these paths are missing:\n"
            f"{formatted}\n"
            "If datasets/processed/eval_dev_all.jsonl is missing, run:\n"
            "uv run python -m src.data.run_pipeline\n"
            "uv run python -m src.data.export_eval"
        )


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    output_path = Path(args.output_path)
    zip_root = Path(args.zip_root)

    required_paths = [
        project_root / "pyproject.toml",
        project_root / "uv.lock",
        project_root / "README.md",
        project_root / "src",
        project_root / args.train_file,
        project_root / args.eval_file,
        project_root / args.eval_set_file,
    ]
    if args.include_docs:
        required_paths.append(project_root / "docs")

    require_existing(required_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        add_file(zip_handle, project_root / "pyproject.toml", zip_root / "pyproject.toml")
        add_file(zip_handle, project_root / "uv.lock", zip_root / "uv.lock")
        add_file(zip_handle, project_root / "README.md", zip_root / "README.md")
        add_path(zip_handle, project_root / "src", zip_root / "src")

        if args.include_docs:
            add_path(zip_handle, project_root / "docs", zip_root / "docs")

        data_files = [
            project_root / args.train_file,
            project_root / args.eval_file,
            project_root / args.eval_set_file,
        ]
        for data_file in data_files:
            add_file(zip_handle, data_file, zip_root / data_file.relative_to(project_root))

    print(
        f"Wrote {output_path.as_posix()} with train={args.train_file} "
        f"eval={args.eval_file} and eval_set={args.eval_set_file}"
    )


if __name__ == "__main__":
    main()
