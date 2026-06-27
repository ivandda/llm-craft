import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.data.quality import QualityConfig, normalize_concept, score_concept


DEFAULT_PROMPT_VERSION = "teacher_structured_enrichment_v2"
DEFAULT_OUTPUT_DATASET_NAME = "dataset_03_teacher_structured_enriched"
SPLITS = ("train", "dev", "test")
PROMPT_STYLES = ("balanced", "strict")
MODEL_PRICE_BY_MILLION_TOKENS = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}


class TeacherCallError(RuntimeError):
    pass


class TeacherCandidate(BaseModel):
    output: str = Field(description="A short concept produced by combining the inputs.")
    rationale: str = Field(
        description="One short visible explanation for why this output follows from the inputs."
    )
    source: Literal["observed", "teacher"] = Field(
        description="Use observed only for outputs copied exactly from the observed output list; use teacher for new alternatives."
    )


class TeacherEnrichment(BaseModel):
    keep_recipe: bool = Field(
        description="False only when the observed recipe is clearly malformed or unrelated."
    )
    reject_reason: str | None = Field(
        default=None,
        description="Short reason used only when keep_recipe is false.",
    )
    candidate_outputs: list[TeacherCandidate] = Field(
        default_factory=list,
        description=(
            "Accepted outputs ordered from strongest to weakest recipe. Include both kept observed outputs "
            "and optional new alternatives."
        ),
    )


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class TeacherResult:
    enrichment: TeacherEnrichment
    usage: TokenUsage = TokenUsage()


class Teacher(Protocol):
    def generate(
        self,
        input_a: str,
        input_b: str,
        observed_outputs: list[str],
        max_alternatives: int,
    ) -> TeacherResult:
        ...


@dataclass(frozen=True)
class StructuredEnrichmentConfig:
    provider: str = "google_vertex_ai"
    model: str = "gemini-2.5-flash-lite"
    location: str = "us-central1"
    input_dataset_name: str = "recipes"
    output_dataset_name: str = DEFAULT_OUTPUT_DATASET_NAME
    target_num_outputs: int = 5
    prompt_version: str = DEFAULT_PROMPT_VERSION
    prompt_style: str = "balanced"
    max_retries: int = 2
    temperature: float = 0.4
    thinking_level: str | None = None
    thinking_budget: int | None = None
    max_rationale_words: int = 24
    input_price_per_million_tokens: float = 0.10
    output_price_per_million_tokens: float = 0.40


class VertexStructuredTeacher:
    def __init__(self, config: StructuredEnrichmentConfig, repo_root: Path):
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

        model = ChatGoogleGenerativeAI(
            model=config.model,
            vertexai=True,
            project=project_id,
            location=config.location,
            temperature=config.temperature,
            thinking_level=config.thinking_level,
            thinking_budget=config.thinking_budget,
        )
        self.config = config
        self.model = model.with_structured_output(
            TeacherEnrichment,
            method="json_schema",
            include_raw=True,
        )

    def generate(
        self,
        input_a: str,
        input_b: str,
        observed_outputs: list[str],
        max_alternatives: int,
    ) -> TeacherResult:
        prompt = build_teacher_prompt(
            input_a=input_a,
            input_b=input_b,
            observed_outputs=observed_outputs,
            max_alternatives=max_alternatives,
            max_rationale_words=self.config.max_rationale_words,
            prompt_style=self.config.prompt_style,
        )
        response = self.model.invoke(prompt)
        parsing_error = response.get("parsing_error")
        if parsing_error is not None:
            raise TeacherCallError(f"Structured response parse failed: {parsing_error}")
        parsed = response.get("parsed")
        if not isinstance(parsed, TeacherEnrichment):
            raise TeacherCallError("Structured response did not return TeacherEnrichment.")
        return TeacherResult(enrichment=parsed, usage=extract_token_usage(response.get("raw")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-call teacher enrichment with structured outputs, alternatives, and rationales."
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
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many records per split instead of reading the first records.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Seed used with --sample-size.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override structured_enrichment.model from config.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default=None,
        help="Override structured_enrichment.prompt_style from config.",
    )
    parser.add_argument(
        "--output-dataset-name",
        default=None,
        help="Override structured_enrichment.output_dataset_name from config.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Override structured_enrichment.thinking_level for supported Gemini models.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Override structured_enrichment.thinking_budget for supported Gemini models.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows that already have at least one candidate output.",
    )
    return parser.parse_args()


def build_teacher_prompt(
    input_a: str,
    input_b: str,
    observed_outputs: list[str],
    max_alternatives: int,
    max_rationale_words: int,
    prompt_style: str = "balanced",
) -> str:
    observed_json = json.dumps(observed_outputs, ensure_ascii=False)
    base_prompt = (
        "You enrich an Infinite Craft style recipe dataset.\n"
        "A valid output is a plausible, creative result of combining Input A and Input B.\n"
        "The rationale must explain how both inputs contribute to the output.\n"
        "Reject outputs where only one input explains the result.\n"
        "Funny, symbolic, meme-like, cultural, or wordplay answers are allowed when the link is clear.\n"
        "Reject malformed text, noisy nonsense, identity copies, near-copies, arbitrary strings, "
        "generic background words, and outputs with no reasonable relation to both inputs.\n"
        "Do not rescue a weak observed output by inventing a long story. Omit it instead.\n"
        "Prefer direct blends, transformations, materials, creatures, places, events, or concepts "
        "that clearly use both inputs.\n"
        "Keep good observed outputs by copying their output string exactly from Observed outputs.\n"
        "Add new alternatives only when they are genuinely plausible. Do not force answers.\n"
        f"Return at most {max_alternatives} new alternative outputs.\n"
        "Return all accepted candidates in candidate_outputs, ordered from strongest to weakest recipe.\n"
        "Strongest means most direct, plausible, specific, and useful as a training target.\n"
        "Do not use numeric scores.\n"
        "Set source='observed' only for outputs copied exactly from Observed outputs. Set source='teacher' for new alternatives.\n"
        f"Each output should be a short concept. Each rationale must be at most {max_rationale_words} words.\n"
        "Good example: fire + water -> steam, because fire heats water into vapor.\n"
        "Bad example: animal + cloud -> rainbow, because cloud explains rainbow but animal does not.\n"
    )
    if prompt_style == "strict":
        base_prompt += (
            "\nUse this decision process before returning any output:\n"
            "1. Can a short rationale explain both inputs without a forced story?\n"
            "2. Is the output more specific than a generic place, object, or category?\n"
            "3. Would the output still make sense if either input were removed? If yes, reject it.\n"
            "4. If unsure, omit the output. A partial record is better than a bad answer.\n\n"
            "Accept example: bee + flower -> honey, because bees gather nectar from flowers to make honey.\n"
            "Accept example: sword + fire -> flaming sword, because fire transforms the sword into a burning weapon.\n"
            "Reject example: cat + ocean -> fish, because ocean explains fish but cat does not contribute.\n"
            "Reject example: robot + mountain -> sky, because sky is generic background, not a combination.\n"
        )
    return (
        base_prompt
        + "\n"
        f"Input A: {input_a}\n"
        f"Input B: {input_b}\n"
        f"Observed outputs: {observed_json}\n"
    )


def enrich_recipe(
    recipe: dict,
    split: str,
    teacher: Teacher,
    config: StructuredEnrichmentConfig,
) -> tuple[dict | None, dict | None, TokenUsage]:
    input_a = normalize_concept(recipe.get("input_a", ""))
    input_b = normalize_concept(recipe.get("input_b", ""))
    observed_outputs = normalize_valid_outputs(
        recipe.get("outputs", []),
        input_a=input_a,
        input_b=input_b,
        existing_outputs=[],
        limit=config.target_num_outputs,
    )

    if not input_a or not input_b or not observed_outputs:
        return None, build_rejection(
            input_a=input_a,
            input_b=input_b,
            split=split,
            reason="no_valid_observed_outputs",
            detail=None,
            outputs=recipe.get("outputs", []),
        ), TokenUsage()

    max_alternatives = max(config.target_num_outputs - len(observed_outputs), 0)
    last_error = None
    usage = TokenUsage()
    teacher_enrichment = None
    for _attempt in range(config.max_retries + 1):
        try:
            result = teacher.generate(input_a, input_b, observed_outputs, max_alternatives)
            usage = usage.add(result.usage)
            teacher_enrichment = result.enrichment
            break
        except TeacherCallError as exc:
            last_error = str(exc)

    if teacher_enrichment is None:
        return None, build_rejection(
            input_a=input_a,
            input_b=input_b,
            split=split,
            reason="teacher_response_error",
            detail=last_error,
            outputs=observed_outputs,
        ), usage

    if not teacher_enrichment.keep_recipe:
        return None, build_rejection(
            input_a=input_a,
            input_b=input_b,
            split=split,
            reason=normalize_reject_reason(teacher_enrichment.reject_reason),
            detail=None,
            outputs=observed_outputs,
        ), usage

    record = build_enriched_record(
        input_a=input_a,
        input_b=input_b,
        observed_outputs=observed_outputs,
        teacher_enrichment=teacher_enrichment,
        split=split,
        config=config,
        usage=usage,
    )
    if not record["candidate_outputs"]:
        return None, build_rejection(
            input_a=input_a,
            input_b=input_b,
            split=split,
            reason="no_valid_teacher_candidates",
            detail=None,
            outputs=observed_outputs,
        ), usage

    return record, None, usage


def build_enriched_record(
    input_a: str,
    input_b: str,
    observed_outputs: list[str],
    teacher_enrichment: TeacherEnrichment,
    split: str,
    config: StructuredEnrichmentConfig,
    usage: TokenUsage,
) -> dict:
    observed_set = set(observed_outputs)
    seen: set[str] = set()
    seen_keys: set[str] = set()
    candidates = []

    for candidate in teacher_enrichment.candidate_outputs:
        output = normalize_concept(candidate.output)
        dedupe_key = concept_dedupe_key(output)
        rationale = normalize_rationale(candidate.rationale)
        source = candidate.source
        if source == "observed" and output not in observed_set:
            continue
        if source == "teacher" and output in observed_set:
            source = "observed"
        if (
            not output
            or not rationale
            or output in seen
            or dedupe_key in seen_keys
            or output == input_a
            or output == input_b
            or not score_concept(output, QualityConfig(reject_gerund_phrases=False)).keep
        ):
            continue
        candidates.append({"output": output, "source": source, "rationale": rationale})
        seen.add(output)
        seen_keys.add(dedupe_key)
        if len(candidates) >= config.target_num_outputs:
            break

    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    quality_status = "complete" if len(candidates) >= config.target_num_outputs else "partial_enrichment"
    return {
        "input_a": input_a,
        "input_b": input_b,
        "candidate_outputs": candidates[: config.target_num_outputs],
        "quality_status": quality_status,
        "metadata": {
            "source_dataset": config.input_dataset_name,
            "source_split": split,
            "teacher_provider": config.provider,
            "teacher_model": config.model,
            "enrichment_version": config.prompt_version,
            "target_num_outputs": config.target_num_outputs,
            "has_rationales": True,
            "has_ranked_outputs": True,
            "partial_outputs_allowed": True,
            "token_usage": asdict(usage),
        },
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


def normalize_rationale(value: str) -> str:
    return " ".join(value.strip().split())


def concept_dedupe_key(value: str) -> str:
    return normalize_concept(value).replace(" ", "").replace("-", "").replace("'", "")


def normalize_reject_reason(value: str | None) -> str:
    reason = normalize_concept(value or "")
    if not reason:
        return "teacher_rejected_recipe"
    return reason.replace(" ", "_")[:80]


def build_rejection(
    input_a: str,
    input_b: str,
    split: str,
    reason: str,
    detail: str | None,
    outputs: list[str],
) -> dict:
    return {
        "input_a": input_a,
        "input_b": input_b,
        "outputs": outputs,
        "split": split,
        "reject_reason": reason,
        "detail": detail,
    }


def extract_token_usage(raw_message) -> TokenUsage:
    raw_usage = getattr(raw_message, "usage_metadata", None) or {}
    return TokenUsage(
        input_tokens=int(raw_usage.get("input_tokens", 0) or 0),
        output_tokens=int(raw_usage.get("output_tokens", 0) or 0),
        total_tokens=int(raw_usage.get("total_tokens", 0) or 0),
    )


def estimate_cost(usage: TokenUsage, config: StructuredEnrichmentConfig) -> float:
    return (
        usage.input_tokens * config.input_price_per_million_tokens
        + usage.output_tokens * config.output_price_per_million_tokens
    ) / 1_000_000


def load_config(repo_root: Path) -> StructuredEnrichmentConfig:
    config_path = repo_root / "configs" / "pipeline_config.yaml"
    if not config_path.exists():
        return StructuredEnrichmentConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    raw_enrichment = raw_config.get("structured_enrichment", {})
    allowed_fields = set(StructuredEnrichmentConfig.__dataclass_fields__)
    filtered = {key: value for key, value in raw_enrichment.items() if key in allowed_fields}
    config = StructuredEnrichmentConfig(**filtered)
    if config.prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt_style: {config.prompt_style}")
    return config


def apply_arg_overrides(
    config: StructuredEnrichmentConfig,
    args: argparse.Namespace,
) -> StructuredEnrichmentConfig:
    overrides = {}
    if args.model:
        overrides["model"] = args.model
        if args.model in MODEL_PRICE_BY_MILLION_TOKENS:
            input_price, output_price = MODEL_PRICE_BY_MILLION_TOKENS[args.model]
            overrides["input_price_per_million_tokens"] = input_price
            overrides["output_price_per_million_tokens"] = output_price
    if args.prompt_style:
        overrides["prompt_style"] = args.prompt_style
    if args.output_dataset_name:
        overrides["output_dataset_name"] = args.output_dataset_name
    if args.thinking_level:
        overrides["thinking_level"] = args.thinking_level
    if args.thinking_budget is not None:
        overrides["thinking_budget"] = args.thinking_budget
    if not overrides:
        return config
    return replace(config, **overrides)


def enrich_split(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    split: str,
    teacher: Teacher,
    config: StructuredEnrichmentConfig,
    limit: int | None,
    resume: bool,
    sample_size: int | None = None,
    seed: int = 13,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    completed_keys = read_completed_keys(output_path) if resume else set()
    mode = "a" if resume and output_path.exists() else "w"
    counts = {
        "read": 0,
        "written": 0,
        "skipped_existing": 0,
        "complete": 0,
        "partial_enrichment": 0,
        "rejected": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    recipes = load_recipes(input_path, sample_size=sample_size, seed=seed)

    with output_path.open(mode, encoding="utf-8") as target, rejected_path.open("a", encoding="utf-8") as rejected:
        for recipe in recipes:
            key = recipe_key(recipe)
            if key in completed_keys:
                counts["skipped_existing"] += 1
                continue

            record, rejection, usage = enrich_recipe(recipe, split, teacher, config)
            counts["read"] += 1
            counts["input_tokens"] += usage.input_tokens
            counts["output_tokens"] += usage.output_tokens
            counts["total_tokens"] += usage.total_tokens
            if record is not None:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["written"] += 1
                counts[record["quality_status"]] += 1
            if rejection is not None:
                rejected.write(json.dumps(rejection, ensure_ascii=False) + "\n")
                counts["rejected"] += 1

            if limit is not None and counts["read"] >= limit:
                break

    return counts


def load_recipes(input_path: Path, sample_size: int | None, seed: int) -> list[dict]:
    recipes = []
    rng = random.Random(seed)
    seen_count = 0
    with input_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            seen_count += 1
            recipe = json.loads(line)
            if sample_size is None:
                recipes.append(recipe)
                continue
            if len(recipes) < sample_size:
                recipes.append(recipe)
            else:
                replacement_index = rng.randrange(seen_count)
                if replacement_index < sample_size:
                    recipes[replacement_index] = recipe
    return recipes


def read_completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            candidates = record.get("candidate_outputs", [])
            if candidates:
                keys.add((record.get("input_a", ""), record.get("input_b", "")))
    return keys


def recipe_key(recipe: dict) -> tuple[str, str]:
    return normalize_concept(recipe.get("input_a", "")), normalize_concept(recipe.get("input_b", ""))


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = apply_arg_overrides(load_config(repo_root), args)

    processed_dir = repo_root / "datasets" / "processed"
    output_dir = repo_root / "datasets" / "enriched" / config.output_dataset_name
    rejected_path = output_dir / "rejected.jsonl"

    teacher = VertexStructuredTeacher(config, repo_root)
    if not args.resume and rejected_path.exists():
        rejected_path.unlink()

    split_counts = {}
    total_usage = TokenUsage()
    for split in args.splits:
        input_path = processed_dir / f"recipes_{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Recipe split not found: {input_path}")
        counts = enrich_split(
            input_path=input_path,
            output_path=output_dir / f"{split}.jsonl",
            rejected_path=rejected_path,
            split=split,
            teacher=teacher,
            config=config,
            limit=args.limit,
            resume=args.resume,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        split_counts[split] = counts
        total_usage = total_usage.add(
            TokenUsage(
                input_tokens=counts["input_tokens"],
                output_tokens=counts["output_tokens"],
                total_tokens=counts["total_tokens"],
            )
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "splits": split_counts,
        "limit": args.limit,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "token_usage": asdict(total_usage),
        "estimated_cost_usd": round(estimate_cost(total_usage, config), 6),
        "input_files": [str(processed_dir / f"recipes_{split}.jsonl") for split in args.splits],
    }
    write_manifest(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
