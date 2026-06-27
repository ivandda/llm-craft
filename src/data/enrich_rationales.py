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


DEFAULT_RATIONALE_VERSION = "teacher_multi_output_with_rationale_v1"
DEFAULT_INPUT_DATASET_NAME = "dataset_01_teacher_enriched_multi_output_no_rationale"
DEFAULT_OUTPUT_DATASET_NAME = "dataset_02_teacher_enriched_multi_output_with_rationale"
SPLITS = ("train", "dev", "test")


class TeacherResponseError(ValueError):
    pass


class RationaleTeacher(Protocol):
    def generate(
        self,
        input_a: str,
        input_b: str,
        candidates: list[dict],
        max_rationale_words: int,
    ) -> str:
        ...


@dataclass(frozen=True)
class RationaleConfig:
    provider: str = "google_vertex_ai"
    model: str = "gemini-2.5-flash"
    location: str = "us-central1"
    input_dataset_name: str = DEFAULT_INPUT_DATASET_NAME
    output_dataset_name: str = DEFAULT_OUTPUT_DATASET_NAME
    prompt_version: str = DEFAULT_RATIONALE_VERSION
    rationale_language: str = "en"
    max_retries: int = 2
    temperature: float = 0.4
    max_rationale_words: int = 24


class VertexGeminiRationaleTeacher:
    def __init__(self, config: RationaleConfig, repo_root: Path):
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
        candidates: list[dict],
        max_rationale_words: int,
    ) -> str:
        prompt = build_teacher_prompt(input_a, input_b, candidates, max_rationale_words)
        response = self.model.invoke(prompt)
        return _message_content_to_text(response.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add concise teacher rationales to an existing multi-output recipe dataset."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Dataset splits to enrich with rationales.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum records per split for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows already present in the output split file with all rationales.",
    )
    return parser.parse_args()


def build_teacher_prompt(
    input_a: str,
    input_b: str,
    candidates: list[dict],
    max_rationale_words: int,
) -> str:
    outputs_json = json.dumps(
        [{"output": candidate.get("output", "")} for candidate in candidates],
        ensure_ascii=False,
    )
    return (
        "You explain Infinite Craft style recipe combinations.\n"
        "Return only strict JSON with the shape "
        "{\"rationales\": [{\"output\": \"...\", \"rationale\": \"...\"}]}.\n"
        "Do not use markdown, numbering, comments, or extra keys.\n"
        "Do not change, add, remove, reorder, translate, or normalize any output string.\n"
        "Do not provide step-by-step reasoning or hidden chain-of-thought.\n"
        "For each output, write one brief visible rationale explaining the compositional "
        "connection between Input A, Input B, and the output.\n"
        "A rationale may use a physical explanation, conceptual analogy, cultural association, "
        "or wordplay when appropriate.\n"
        f"Each rationale must be at most {max_rationale_words} words.\n\n"
        f"Input A: {input_a}\n"
        f"Input B: {input_b}\n"
        f"Outputs, in required order: {outputs_json}\n"
    )


def parse_teacher_rationales(raw_response: str, expected_outputs: list[str]) -> dict[str, str]:
    raw_response = _strip_json_fence(raw_response)
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise TeacherResponseError("Teacher response is not valid JSON.") from exc

    rationales = parsed.get("rationales") if isinstance(parsed, dict) else None
    if not isinstance(rationales, list):
        raise TeacherResponseError("Teacher response must contain a list under `rationales`.")

    expected_set = set(expected_outputs)
    seen: set[str] = set()
    result: dict[str, str] = {}
    for index, item in enumerate(rationales):
        if not isinstance(item, dict):
            raise TeacherResponseError("Each rationale item must be an object.")
        output = item.get("output")
        rationale = item.get("rationale")
        if not isinstance(output, str) or not isinstance(rationale, str):
            raise TeacherResponseError(
                "Each rationale item must contain string `output` and `rationale`."
            )
        if output not in expected_set:
            raise TeacherResponseError(f"Teacher returned unexpected output: {output}")
        if output in seen:
            raise TeacherResponseError(f"Teacher returned duplicate output: {output}")
        if index < len(expected_outputs) and output != expected_outputs[index]:
            raise TeacherResponseError("Teacher changed the required output order.")

        stripped_rationale = " ".join(rationale.split())
        if stripped_rationale:
            result[output] = stripped_rationale
        seen.add(output)

    return result


def enrich_record_with_rationales(
    source_record: dict,
    split: str,
    teacher: RationaleTeacher,
    config: RationaleConfig,
) -> tuple[dict, dict | None]:
    input_a = source_record.get("input_a", "")
    input_b = source_record.get("input_b", "")
    candidates = [dict(candidate) for candidate in source_record.get("candidate_outputs", [])]
    expected_outputs = [candidate.get("output", "") for candidate in candidates]

    rationale_by_output: dict[str, str] = {
        candidate["output"]: candidate["rationale"]
        for candidate in candidates
        if isinstance(candidate.get("output"), str)
        and isinstance(candidate.get("rationale"), str)
        and candidate["rationale"].strip()
    }
    missing_outputs = [output for output in expected_outputs if output not in rationale_by_output]
    last_error = None

    for _attempt in range(config.max_retries + 1):
        if not missing_outputs:
            break
        try:
            raw_response = teacher.generate(input_a, input_b, candidates, config.max_rationale_words)
            parsed_rationales = parse_teacher_rationales(raw_response, expected_outputs)
            rationale_by_output.update(parsed_rationales)
            missing_outputs = [output for output in expected_outputs if output not in rationale_by_output]
            if not missing_outputs:
                break
        except TeacherResponseError as exc:
            last_error = str(exc)

    record = build_rationale_record(source_record, rationale_by_output, split, config)
    if record["quality_status"] == "complete":
        return record, None

    has_any_rationale = any("rationale" in candidate for candidate in record["candidate_outputs"])
    reason = "missing_rationales" if has_any_rationale else "teacher_response_error"
    if not has_any_rationale and last_error:
        record["quality_status"] = "failed_rationale"
    return record, {
        "input_a": input_a,
        "input_b": input_b,
        "split": split,
        "reason": reason,
        "detail": last_error,
        "missing_outputs": [
            candidate.get("output", "")
            for candidate in record["candidate_outputs"]
            if not candidate.get("rationale")
        ],
    }


def build_rationale_record(
    source_record: dict,
    rationale_by_output: dict[str, str],
    split: str,
    config: RationaleConfig,
) -> dict:
    candidates = []
    for source_candidate in source_record.get("candidate_outputs", []):
        candidate = {
            "output": source_candidate.get("output", ""),
            "source": source_candidate.get("source", ""),
        }
        rationale = rationale_by_output.get(candidate["output"])
        if rationale:
            candidate["rationale"] = rationale
        candidates.append(candidate)

    rationale_count = sum(1 for candidate in candidates if candidate.get("rationale"))
    if candidates and rationale_count == len(candidates):
        quality_status = "complete"
    elif rationale_count > 0 or candidates:
        quality_status = "partial_rationale"
    else:
        quality_status = "failed_rationale"

    source_metadata = source_record.get("metadata", {})
    target_num_outputs = source_metadata.get("target_num_outputs", len(candidates))
    return {
        "input_a": source_record.get("input_a", ""),
        "input_b": source_record.get("input_b", ""),
        "candidate_outputs": candidates,
        "quality_status": quality_status,
        "metadata": {
            "source_dataset": config.input_dataset_name,
            "source_split": split,
            "teacher_provider": config.provider,
            "teacher_model": config.model,
            "enrichment_version": config.prompt_version,
            "target_num_outputs": target_num_outputs,
            "has_rationales": True,
            "rationale_language": config.rationale_language,
        },
    }


def load_config(repo_root: Path) -> RationaleConfig:
    config_path = repo_root / "configs" / "pipeline_config.yaml"
    if not config_path.exists():
        return RationaleConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    raw_rationale_config = raw_config.get("rationale_enrichment", {})
    allowed_fields = set(RationaleConfig.__dataclass_fields__)
    filtered = {key: value for key, value in raw_rationale_config.items() if key in allowed_fields}
    return RationaleConfig(**filtered)


def enrich_split(
    input_path: Path,
    output_path: Path,
    failed_path: Path,
    split: str,
    teacher: RationaleTeacher,
    config: RationaleConfig,
    limit: int | None,
    resume: bool,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    completed_keys = read_completed_rationale_keys(output_path) if resume else set()
    mode = "a" if resume and output_path.exists() else "w"
    counts = {
        "read": 0,
        "written": 0,
        "skipped_existing": 0,
        "complete": 0,
        "partial_rationale": 0,
        "failed_rationale": 0,
        "failures": 0,
    }

    with (
        input_path.open("r", encoding="utf-8") as source,
        output_path.open(mode, encoding="utf-8") as target,
        failed_path.open("a", encoding="utf-8") as failures,
    ):
        for line in source:
            if not line.strip():
                continue
            source_record = json.loads(line)
            key = record_key(source_record)
            if key in completed_keys:
                counts["skipped_existing"] += 1
                continue

            record, failure = enrich_record_with_rationales(
                source_record=source_record,
                split=split,
                teacher=teacher,
                config=config,
            )
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


def read_completed_rationale_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = record.get("candidate_outputs", [])
            if candidates and all(candidate.get("rationale") for candidate in candidates):
                keys.add(record_key(record))
    return keys


def record_key(record: dict) -> tuple[str, str]:
    return record.get("input_a", ""), record.get("input_b", "")


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

    enriched_dir = repo_root / "datasets" / "enriched"
    input_dir = enriched_dir / config.input_dataset_name
    output_dir = enriched_dir / config.output_dataset_name
    failed_path = output_dir / "failed_generations.jsonl"

    teacher = VertexGeminiRationaleTeacher(config, repo_root)
    if not args.resume and failed_path.exists():
        failed_path.unlink()

    split_counts = {}
    for split in args.splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Enriched source split not found: {input_path}")

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
        "limit": args.limit,
        "input_files": [str(input_dir / f"{split}.jsonl") for split in args.splits],
    }
    write_manifest(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
