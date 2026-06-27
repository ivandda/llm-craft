import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import yaml
from dotenv import load_dotenv

from src.data.quality import QualityConfig, normalize_concept, score_concept


DEFAULT_ENRICHMENT_VERSION = "teacher_multi_output_no_rationale_v1"
DEFAULT_OUTPUT_DATASET_NAME = "dataset_01_teacher_enriched_multi_output_no_rationale"
BASELINE_DATASET_NAME = "dataset_00_recipes_baseline_multi_output"
SPLITS = ("train", "dev", "test")


class TeacherResponseError(ValueError):
    pass


class Teacher(Protocol):
    def generate(
        self,
        input_a: str,
        input_b: str,
        existing_outputs: list[str],
        num_outputs: int,
    ) -> str:
        ...


@dataclass(frozen=True)
class EnrichmentConfig:
    provider: str = "google_vertex_ai"
    model: str = "gemini-2.5-flash"
    location: str = "us-central1"
    input_dataset_name: str = "recipes"
    output_dataset_name: str = DEFAULT_OUTPUT_DATASET_NAME
    target_num_outputs: int = 5
    prompt_version: str = DEFAULT_ENRICHMENT_VERSION
    max_retries: int = 2
    temperature: float = 0.7


class VertexGeminiTeacher:
    def __init__(self, config: EnrichmentConfig, repo_root: Path):
        from langchain_google_genai import ChatGoogleGenerativeAI

        env_path = repo_root / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        creds_rel_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_rel_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((repo_root / creds_rel_path).resolve())

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set in environment or .env file.")

        self.config = config
        self.model = ChatGoogleGenerativeAI(
            model=config.model,
            vertexai=True,
            project=project_id,
            location=config.location,
            temperature=config.temperature,
        )

    def generate(
        self,
        input_a: str,
        input_b: str,
        existing_outputs: list[str],
        num_outputs: int,
    ) -> str:
        prompt = build_teacher_prompt(input_a, input_b, existing_outputs, num_outputs)
        response = self.model.invoke(prompt)
        return _message_content_to_text(response.content)


class DryRunTeacher:
    def generate(
        self,
        input_a: str,
        input_b: str,
        existing_outputs: list[str],
        num_outputs: int,
    ) -> str:
        return json.dumps({"outputs": []})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a grouped multi-output recipe dataset enriched with a teacher model."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Recipe splits to enrich.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum records per split for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the teacher; write observed outputs only and mark incomplete rows as partial.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows already present in the output split file.",
    )
    return parser.parse_args()


def build_teacher_prompt(
    input_a: str,
    input_b: str,
    existing_outputs: list[str],
    num_outputs: int,
) -> str:
    existing = ", ".join(existing_outputs)
    return (
        "You generate concise alternative results for an Infinite Craft style recipe.\n"
        "Return only strict JSON with the shape {\"outputs\": [\"...\"]}.\n"
        "Do not include rationales, explanations, markdown, numbering, or extra keys.\n"
        f"Generate exactly {num_outputs} new plausible outputs.\n"
        "Each output must be a short concept of at most three words.\n"
        "Do not repeat existing outputs and do not return either input unchanged.\n\n"
        f"Input A: {input_a}\n"
        f"Input B: {input_b}\n"
        f"Existing outputs: {existing}\n"
    )


def parse_teacher_outputs(raw_response: str) -> list[str]:
    raw_response = _strip_json_fence(raw_response)
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise TeacherResponseError("Teacher response is not valid JSON.") from exc

    outputs = parsed.get("outputs") if isinstance(parsed, dict) else None
    if not isinstance(outputs, list) or not all(isinstance(output, str) for output in outputs):
        raise TeacherResponseError("Teacher response must contain a string list under `outputs`.")
    return outputs


def build_enriched_record(
    recipe: dict,
    teacher_outputs: list[str],
    split: str,
    config: EnrichmentConfig,
) -> dict:
    input_a = normalize_concept(recipe.get("input_a", ""))
    input_b = normalize_concept(recipe.get("input_b", ""))
    observed_outputs = normalize_valid_outputs(
        recipe.get("outputs", []),
        input_a=input_a,
        input_b=input_b,
        existing_outputs=[],
        limit=config.target_num_outputs,
    )
    remaining = config.target_num_outputs - len(observed_outputs)
    teacher_valid_outputs = normalize_valid_outputs(
        teacher_outputs,
        input_a=input_a,
        input_b=input_b,
        existing_outputs=observed_outputs,
        limit=max(remaining, 0),
    )

    candidate_outputs = [
        {"output": output, "source": "observed"} for output in observed_outputs
    ] + [
        {"output": output, "source": "teacher"} for output in teacher_valid_outputs
    ]

    if len(candidate_outputs) >= config.target_num_outputs:
        quality_status = "complete"
    elif candidate_outputs:
        quality_status = "partial_enrichment"
    else:
        quality_status = "failed_enrichment"

    return {
        "input_a": input_a,
        "input_b": input_b,
        "candidate_outputs": candidate_outputs[: config.target_num_outputs],
        "quality_status": quality_status,
        "metadata": {
            "source_dataset": config.input_dataset_name,
            "source_split": split,
            "teacher_provider": config.provider,
            "teacher_model": config.model,
            "enrichment_version": config.prompt_version,
            "target_num_outputs": config.target_num_outputs,
            "has_rationales": False,
        },
    }


def enrich_recipe(
    recipe: dict,
    split: str,
    teacher: Teacher,
    config: EnrichmentConfig,
) -> tuple[dict, dict | None]:
    input_a = normalize_concept(recipe.get("input_a", ""))
    input_b = normalize_concept(recipe.get("input_b", ""))
    observed_outputs = normalize_valid_outputs(
        recipe.get("outputs", []),
        input_a=input_a,
        input_b=input_b,
        existing_outputs=[],
        limit=config.target_num_outputs,
    )

    missing_count = config.target_num_outputs - len(observed_outputs)
    if missing_count <= 0:
        return build_enriched_record(recipe, [], split, config), None

    teacher_outputs: list[str] = []
    last_error = None
    for attempt in range(config.max_retries + 1):
        needed = missing_count - len(teacher_outputs)
        if needed <= 0:
            break
        try:
            raw_response = teacher.generate(input_a, input_b, observed_outputs + teacher_outputs, needed)
            parsed_outputs = parse_teacher_outputs(raw_response)
            teacher_outputs.extend(
                normalize_valid_outputs(
                    parsed_outputs,
                    input_a=input_a,
                    input_b=input_b,
                    existing_outputs=observed_outputs + teacher_outputs,
                    limit=needed,
                )
            )
            if len(teacher_outputs) >= missing_count:
                break
        except TeacherResponseError as exc:
            last_error = str(exc)

    record = build_enriched_record(recipe, teacher_outputs, split, config)
    if record["quality_status"] == "complete":
        return record, None

    reason = "insufficient_valid_teacher_outputs"
    if last_error and not teacher_outputs:
        reason = "teacher_response_error"

    return record, {
        "input_a": input_a,
        "input_b": input_b,
        "split": split,
        "reason": reason,
        "detail": last_error,
        "observed_output_count": len(observed_outputs),
        "candidate_output_count": len(record["candidate_outputs"]),
    }


def normalize_valid_outputs(
    outputs: list[str],
    input_a: str,
    input_b: str,
    existing_outputs: list[str],
    limit: int,
    quality_config: QualityConfig | None = None,
) -> list[str]:
    quality_config = quality_config or QualityConfig(reject_gerund_phrases=False)
    seen = set(existing_outputs)
    normalized_outputs = []

    for output in outputs:
        normalized = normalize_concept(output)
        if (
            not normalized
            or normalized in seen
            or normalized == input_a
            or normalized == input_b
            or not score_concept(normalized, quality_config).keep
        ):
            continue
        normalized_outputs.append(normalized)
        seen.add(normalized)
        if len(normalized_outputs) >= limit:
            break

    return normalized_outputs


def load_config(repo_root: Path) -> EnrichmentConfig:
    config_path = repo_root / "configs" / "pipeline_config.yaml"
    if not config_path.exists():
        return EnrichmentConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    raw_enrichment = raw_config.get("enrichment", {})
    allowed_fields = set(EnrichmentConfig.__dataclass_fields__)
    filtered = {key: value for key, value in raw_enrichment.items() if key in allowed_fields}
    return EnrichmentConfig(**filtered)


def copy_baseline_split(input_path: Path, output_path: Path, limit: int | None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            target.write(line)
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def enrich_split(
    input_path: Path,
    output_path: Path,
    failed_path: Path,
    split: str,
    teacher: Teacher,
    config: EnrichmentConfig,
    limit: int | None,
    resume: bool,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    completed_keys = read_completed_keys(output_path) if resume else set()
    mode = "a" if resume and output_path.exists() else "w"
    counts = {
        "read": 0,
        "written": 0,
        "skipped_existing": 0,
        "complete": 0,
        "partial_enrichment": 0,
        "failed_enrichment": 0,
        "failures": 0,
    }

    with input_path.open("r", encoding="utf-8") as source, output_path.open(mode, encoding="utf-8") as target, failed_path.open("a", encoding="utf-8") as failures:
        for line in source:
            if not line.strip():
                continue
            recipe = json.loads(line)
            key = recipe_key(recipe)
            if key in completed_keys:
                counts["skipped_existing"] += 1
                continue

            record, failure = enrich_recipe(recipe, split, teacher, config)
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["read"] += 1
            counts["written"] += 1
            counts[record["quality_status"]] += 1

            if failure is not None:
                failures.write(json.dumps(failure, ensure_ascii=False) + "\n")
                counts["failures"] += 1

            if limit is not None and counts["read"] >= limit:
                break

    return counts


def read_completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            keys.add((record.get("input_a", ""), record.get("input_b", "")))
    return keys


def recipe_key(recipe: dict) -> tuple[str, str]:
    return normalize_concept(recipe.get("input_a", "")), normalize_concept(recipe.get("input_b", ""))


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _strip_json_fence(raw_response: str) -> str:
    stripped = raw_response.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(repo_root)

    processed_dir = repo_root / "datasets" / "processed"
    enriched_dir = repo_root / "datasets" / "enriched"
    baseline_dir = enriched_dir / BASELINE_DATASET_NAME
    output_dir = enriched_dir / config.output_dataset_name
    failed_path = output_dir / "failed_generations.jsonl"

    teacher: Teacher = DryRunTeacher() if args.dry_run else VertexGeminiTeacher(config, repo_root)
    if not args.resume and failed_path.exists():
        failed_path.unlink()

    split_counts = {}
    baseline_counts = {}
    for split in args.splits:
        input_path = processed_dir / f"recipes_{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Recipe split not found: {input_path}")

        baseline_counts[split] = copy_baseline_split(
            input_path=input_path,
            output_path=baseline_dir / f"{split}.jsonl",
            limit=args.limit,
        )
        split_counts[split] = enrich_split(
            input_path=input_path,
            output_path=output_dir / f"{split}.jsonl",
            failed_path=failed_path,
            split=split,
            teacher=teacher,
            config=config,
            limit=args.limit,
            resume=args.resume,
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "splits": split_counts,
        "baseline_splits": baseline_counts,
        "dry_run": args.dry_run,
        "limit": args.limit,
        "input_files": [str(processed_dir / f"recipes_{split}.jsonl") for split in args.splits],
    }
    write_manifest(output_dir / "manifest.json", manifest)
    write_manifest(
        baseline_dir / "manifest.json",
        {
            "created_at": manifest["created_at"],
            "source_dataset": config.input_dataset_name,
            "splits": baseline_counts,
            "input_files": manifest["input_files"],
        },
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
