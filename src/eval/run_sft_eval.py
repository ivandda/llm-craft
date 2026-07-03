from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel

from src.eval.creativity import CreativityComponents, compute_creativity_components, min_cosine_distances
from src.eval.embeddings import BaseTextEmbedder, build_text_embedder
from src.eval.vertex_judge import VertexAnthropicNoveltyJudge, load_vertex_environment
from src.sft.collator import render_prefix, render_user_prompt
from src.sft.config import SFTConfig
from src.sft.trainer import build_model_and_tokenizer


@dataclass(frozen=True)
class EvalRecord:
    pair_id: str
    input_a: str
    input_b: str
    canonical_output: str
    known_outputs: list[str]


def _str_to_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT evaluation with creativity metrics.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing config.yaml and adapters.")
    parser.add_argument(
        "--adapter_dir",
        default=None,
        help="Optional adapter directory. Defaults to best_adapter, then final_adapter inside run_dir.",
    )
    parser.add_argument("--eval_file", default="datasets/processed/eval_dev_all.jsonl")
    parser.add_argument(
        "--train_reference_file",
        default=None,
        help="Reference train set used for novelty. Defaults to the train_path from the run config.",
    )
    parser.add_argument("--output_dir", default=None, help="Directory to write predictions and summary.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--lambda_penalty", type=float, default=2.0)
    parser.add_argument("--novelty_method", choices=["vertex_judge", "embedding"], default="vertex_judge")
    parser.add_argument("--novelty_judge_model", default=None)
    parser.add_argument("--novelty_judge_region", default=None)
    parser.add_argument(
        "--embedding_backend",
        choices=["glove", "word2vec", "sentence_embeddings"],
        default="sentence_embeddings",
    )
    parser.add_argument(
        "--embedding_model_path",
        default=None,
        help="Path to a local plain-text embedding file for glove or word2vec backends.",
    )
    parser.add_argument(
        "--sentence_embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path for embedding_backend=sentence_embeddings.",
    )
    parser.add_argument("--eval_bf16", type=_str_to_bool, default=None)
    parser.add_argument("--eval_fp16", type=_str_to_bool, default=None)
    parser.add_argument("--eval_bnb_4bit_compute_dtype", default=None)
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return loaded


def is_gcs_uri(value: str) -> bool:
    return value.startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not is_gcs_uri(uri):
        raise ValueError(f"Expected a gs:// URI, got {uri!r}")
    remainder = uri[len("gs://") :]
    bucket, _, blob_prefix = remainder.partition("/")
    if not bucket:
        raise ValueError(f"Invalid gs:// URI without bucket: {uri!r}")
    return bucket, blob_prefix


def build_gcs_client() -> Any:
    from google.cloud import storage

    return storage.Client()


def log_progress(message: str) -> None:
    print(f"[eval] {message}", flush=True)


def download_gcs_file(uri: str, local_path: Path, client: Any) -> Path:
    log_progress(f"Downloading file from {uri} to {local_path}")
    bucket_name, blob_name = parse_gcs_uri(uri)
    if not blob_name:
        raise ValueError(f"GCS file URI must include an object path: {uri!r}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))
    return local_path


def download_gcs_prefix(uri: str, local_dir: Path, client: Any) -> Path:
    log_progress(f"Downloading directory from {uri} to {local_dir}")
    bucket_name, blob_prefix = parse_gcs_uri(uri)
    prefix = blob_prefix.rstrip("/")
    if not prefix:
        raise ValueError(f"GCS directory URI must include a prefix: {uri!r}")
    bucket = client.bucket(bucket_name)
    found_blob = False
    for blob in client.list_blobs(bucket, prefix=prefix + "/"):
        if blob.name.endswith("/"):
            continue
        found_blob = True
        relative_path = Path(blob.name).relative_to(prefix)
        destination = local_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
    if not found_blob:
        raise FileNotFoundError(f"No objects found under GCS prefix {uri!r}")
    return local_dir


def upload_file_to_gcs(local_path: Path, uri: str, client: Any) -> None:
    log_progress(f"Uploading {local_path.name} to {uri}")
    bucket_name, blob_name = parse_gcs_uri(uri)
    if not blob_name:
        raise ValueError(f"GCS file URI must include an object path: {uri!r}")
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))


def stage_run_dir(run_dir: str, staging_root: Path, client: Any | None) -> Path:
    if not is_gcs_uri(run_dir):
        return Path(run_dir)
    if client is None:
        raise RuntimeError("A GCS client is required to stage a gs:// run_dir.")
    return download_gcs_prefix(run_dir, staging_root / "run_dir", client)


def stage_input_file(path_or_uri: str, staging_root: Path, client: Any | None, label: str) -> Path:
    if not is_gcs_uri(path_or_uri):
        return Path(path_or_uri)
    if client is None:
        raise RuntimeError(f"A GCS client is required to stage gs:// {label}.")
    file_name = Path(parse_gcs_uri(path_or_uri)[1]).name or f"{label}.jsonl"
    return download_gcs_file(path_or_uri, staging_root / file_name, client)


def prepare_output_dir(output_dir: str | None, run_dir: Path, eval_file: str, staging_root: Path) -> tuple[Path, str | None]:
    if output_dir is None:
        return resolve_output_dir(run_dir, Path(eval_file), None), None
    if is_gcs_uri(output_dir):
        return staging_root / "output", output_dir.rstrip("/")
    return Path(output_dir), None


def upload_output_artifacts(local_output_dir: Path, output_uri: str | None, client: Any | None) -> None:
    if output_uri is None:
        return
    if client is None:
        raise RuntimeError("A GCS client is required to upload outputs to gs://.")
    upload_file_to_gcs(local_output_dir / "predictions.jsonl", output_uri + "/predictions.jsonl", client)
    upload_file_to_gcs(local_output_dir / "summary.json", output_uri + "/summary.json", client)


def normalize_eval_record(record: dict[str, Any], line_number: int) -> EvalRecord:
    input_a = record.get("input_a")
    input_b = record.get("input_b")
    if not input_a or not input_b:
        raise ValueError(f"Invalid eval record at line {line_number}: missing input_a/input_b.")

    if isinstance(record.get("known_outputs"), list):
        known_outputs = [str(output) for output in record["known_outputs"] if str(output).strip()]
    elif isinstance(record.get("candidate_outputs"), list):
        known_outputs = [str(row["output"]) for row in record["candidate_outputs"] if row.get("output")]
    elif record.get("canonical_output"):
        known_outputs = [str(record["canonical_output"])]
    else:
        known_outputs = []

    if not known_outputs:
        raise ValueError(f"Invalid eval record at line {line_number}: no known outputs.")

    canonical_output = str(record.get("canonical_output") or known_outputs[0])
    pair_id = str(record.get("pair_id") or f"{input_a}+{input_b}")
    return EvalRecord(
        pair_id=pair_id,
        input_a=str(input_a),
        input_b=str(input_b),
        canonical_output=canonical_output,
        known_outputs=known_outputs,
    )


def load_eval_records(path: str | Path, *, max_examples: int | None = None) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            records.append(normalize_eval_record(json.loads(stripped), line_number))
            if max_examples is not None and len(records) >= max_examples:
                break
    return records


def build_train_reference_map(path: str | Path) -> dict[tuple[str, str], list[str]]:
    reference_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = normalize_eval_record(json.loads(stripped), line_number)
            reference_map[(record.input_a, record.input_b)].extend(record.known_outputs)
    deduped: dict[tuple[str, str], list[str]] = {}
    for pair, outputs in reference_map.items():
        seen = set()
        unique_outputs = []
        for output in outputs:
            normalized = output.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_outputs.append(output)
        deduped[pair] = unique_outputs
    return deduped


def resolve_adapter_dir(run_dir: Path, adapter_dir: str | None) -> Path:
    if adapter_dir is not None:
        return Path(adapter_dir)
    for candidate in [run_dir / "best_adapter", run_dir / "final_adapter"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No adapter directory found under {run_dir}.")


def resolve_output_dir(run_dir: Path, eval_file: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return run_dir / "eval" / eval_file.stem


def render_generation_prompt(record: EvalRecord, config: SFTConfig, tokenizer: Any) -> str:
    if config.prompt_format == "qwen_chat":
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": render_user_prompt(record.input_a, record.input_b)})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return render_prefix(record.input_a, record.input_b)


def decode_generated_text(
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_length: int,
) -> list[str]:
    decoded = tokenizer.batch_decode(sequences[:, prompt_length:], skip_special_tokens=True)
    return [text.strip() for text in decoded]


def build_output_record(eval_record: dict[str, Any], prediction: str) -> dict[str, Any]:
    return {
        "pair_id": eval_record["pair_id"],
        "input_a": eval_record["input_a"],
        "input_b": eval_record["input_b"],
        "prediction": prediction,
        "canonical_output": eval_record["canonical_output"],
        "known_outputs": eval_record["known_outputs"],
    }


def creativity_to_dict(components: CreativityComponents) -> dict[str, float]:
    return {
        "plausibility_distance": components.plausibility_distance,
        "plausibility_score": components.plausibility_score,
        "novelty": components.novelty,
        "diversity_distance": components.diversity_distance,
        "diversity_score": components.diversity_score,
        "local_creativity": components.local_creativity,
    }


def create_prediction_record(
    record: EvalRecord,
    sampled_outputs: list[str],
    creativity: CreativityComponents,
) -> dict[str, Any]:
    top_prediction = sampled_outputs[0] if sampled_outputs else ""
    base_record = build_output_record(
        {
            "pair_id": record.pair_id,
            "input_a": record.input_a,
            "input_b": record.input_b,
            "canonical_output": record.canonical_output,
            "known_outputs": record.known_outputs,
        },
        top_prediction,
    )
    base_record["sampled_outputs"] = sampled_outputs
    base_record["creativity"] = creativity_to_dict(creativity)
    return base_record


def summarize_creativity(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "mean_plausibility_distance": 0.0,
            "mean_plausibility_score": 0.0,
            "mean_novelty": 0.0,
            "mean_diversity_distance": 0.0,
            "mean_diversity_score": 0.0,
            "ccs": 0.0,
        }
    return {
        "mean_plausibility_distance": sum(
            record["creativity"]["plausibility_distance"] for record in records
        )
        / len(records),
        "mean_plausibility_score": sum(
            record["creativity"]["plausibility_score"] for record in records
        )
        / len(records),
        "mean_novelty": sum(record["creativity"]["novelty"] for record in records) / len(records),
        "mean_diversity_distance": sum(
            record["creativity"]["diversity_distance"] for record in records
        )
        / len(records),
        "mean_diversity_score": sum(record["creativity"]["diversity_score"] for record in records)
        / len(records),
        "ccs": sum(record["creativity"]["local_creativity"] for record in records) / len(records),
    }


def build_metric_extreme_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": record["pair_id"],
        "input_a": record["input_a"],
        "input_b": record["input_b"],
        "canonical_output": record["canonical_output"],
        "known_outputs": record["known_outputs"],
        "prediction": record["prediction"],
        "sampled_outputs": record["sampled_outputs"],
        "creativity": record["creativity"],
    }


def summarize_creativity_extremes(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"min_local_creativity_recipe": None, "max_local_creativity_recipe": None}

    min_record = min(records, key=lambda record: record["creativity"]["local_creativity"])
    max_record = max(records, key=lambda record: record["creativity"]["local_creativity"])
    return {
        "min_local_creativity_recipe": build_metric_extreme_record(min_record),
        "max_local_creativity_recipe": build_metric_extreme_record(max_record),
    }


def maybe_move_to_device(model: torch.nn.Module, device_override: str | None) -> torch.nn.Module:
    if device_override is None:
        return model
    return model.to(torch.device(device_override))


def apply_eval_precision_overrides(config: SFTConfig, args: argparse.Namespace) -> SFTConfig:
    if args.eval_bf16 is not None:
        config.bf16 = args.eval_bf16
    if args.eval_fp16 is not None:
        config.fp16 = args.eval_fp16
    if args.eval_bnb_4bit_compute_dtype is not None:
        config.bnb_4bit_compute_dtype = args.eval_bnb_4bit_compute_dtype
    return config


def resolve_vertex_judge_model(args: argparse.Namespace) -> str:
    model = args.novelty_judge_model or os.environ.get("VERTEX_NOVELTY_JUDGE_MODEL")
    if not model:
        raise ValueError(
            "novelty_method='vertex_judge' requires --novelty_judge_model or VERTEX_NOVELTY_JUDGE_MODEL."
        )
    return model


def resolve_vertex_judge_region(args: argparse.Namespace) -> str:
    return (
        args.novelty_judge_region
        or os.environ.get("VERTEX_NOVELTY_JUDGE_REGION")
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or "us-east5"
    )


def build_eval_embedder(args: argparse.Namespace) -> BaseTextEmbedder:
    return build_text_embedder(
        args.embedding_backend,
        word_vector_path=args.embedding_model_path,
        sentence_transformer_model=args.sentence_embedding_model,
        device=args.device,
    )


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path.cwd()
    load_vertex_environment(repo_root)
    log_progress("Starting creativity evaluation pipeline")
    gcs_client = build_gcs_client() if any(
        is_gcs_uri(value)
        for value in [args.run_dir, args.eval_file, args.train_reference_file or "", args.output_dir or ""]
    ) else None
    if gcs_client is not None:
        log_progress("GCS mode enabled via gs:// paths")
    with tempfile.TemporaryDirectory(prefix="llm_craft_eval_") as temp_dir:
        staging_root = Path(temp_dir)
        log_progress(f"Using temporary staging directory {staging_root}")
        run_dir = stage_run_dir(args.run_dir, staging_root, gcs_client)
        log_progress(f"Using run directory {run_dir}")
        run_config = SFTConfig(**load_yaml(run_dir / "config.yaml"))
        run_config = apply_eval_precision_overrides(run_config, args)
        adapter_dir = resolve_adapter_dir(run_dir, args.adapter_dir)
        log_progress(f"Using adapter directory {adapter_dir}")
        eval_file = stage_input_file(args.eval_file, staging_root, gcs_client, "eval_file")
        train_reference_file_arg = args.train_reference_file or run_config.train_path
        train_reference_file = stage_input_file(
            train_reference_file_arg,
            staging_root,
            gcs_client,
            "train_reference_file",
        )
        output_dir, output_uri = prepare_output_dir(args.output_dir, run_dir, str(eval_file), staging_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_progress(f"Resolved eval file to {eval_file}")
        log_progress(f"Resolved train reference file to {train_reference_file}")
        log_progress(f"Writing outputs to {output_uri or output_dir}")

        log_progress(f"Loading base model and tokenizer from {run_config.model_name_or_path}")
        model, tokenizer = build_model_and_tokenizer(run_config, apply_lora=False)
        log_progress("Loading PEFT adapter")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = maybe_move_to_device(model, args.device)
        model.eval()
        log_progress(f"Model ready on device {next(model.parameters()).device}")

        eval_records = load_eval_records(eval_file, max_examples=args.max_examples)
        train_reference_map = build_train_reference_map(train_reference_file)
        log_progress(
            f"Loaded {len(eval_records)} eval recipes and {len(train_reference_map)} train reference pairs"
        )
        log_progress(
            f"Building embedder with backend={args.embedding_backend} model={args.sentence_embedding_model}"
        )
        embedder = build_eval_embedder(args)
        novelty_judge = None
        if args.novelty_method == "vertex_judge":
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set in environment or .env file.")
            log_progress(
                f"Initializing Vertex novelty judge model={resolve_vertex_judge_model(args)} region={resolve_vertex_judge_region(args)}"
            )
            novelty_judge = VertexAnthropicNoveltyJudge(
                project_id=project_id,
                region=resolve_vertex_judge_region(args),
                model=resolve_vertex_judge_model(args),
            )

        prediction_records: list[dict[str, Any]] = []
        log_progress(f"Starting generation for {len(eval_records)} recipes with num_samples={args.num_samples}")
        for index, record in enumerate(eval_records, start=1):
            prompt = render_generation_prompt(record, run_config, tokenizer)
            encoded = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
            generation = model.generate(
                **encoded,
                do_sample=args.num_samples > 1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            sampled_outputs = decode_generated_text(tokenizer, generation, encoded["input_ids"].shape[1])
            if not sampled_outputs:
                sampled_outputs = [""]

            sample_embeddings = embedder.encode(sampled_outputs)
            recipe_candidate_embeddings = embedder.encode(record.known_outputs)
            train_outputs = train_reference_map.get((record.input_a, record.input_b), [])
            if args.novelty_method == "vertex_judge":
                assert novelty_judge is not None
                novelty_result = novelty_judge.score_batch(
                    input_a=record.input_a,
                    input_b=record.input_b,
                    generated_outputs=sampled_outputs,
                    recipe_candidates=record.known_outputs,
                    train_outputs=train_outputs,
                )
                novelty_scores = torch.tensor(
                    [result.novelty_score for result in novelty_result.results],
                    dtype=torch.float32,
                )
            else:
                if not train_outputs:
                    raise ValueError(
                        "novelty_method='embedding' requires at least one train reference for every evaluated pair."
                    )
                train_output_embeddings = embedder.encode(train_outputs)
                novelty_scores = min_cosine_distances(sample_embeddings, train_output_embeddings)

            creativity = compute_creativity_components(
                sample_embeddings,
                recipe_candidate_embeddings,
                novelty_scores,
                alpha=args.alpha,
                lambda_penalty=args.lambda_penalty,
            )
            prediction_records.append(create_prediction_record(record, sampled_outputs, creativity))
            if index == 1 or index % 10 == 0 or index == len(eval_records):
                log_progress(f"Processed {index}/{len(eval_records)} recipes")

        summary = summarize_creativity(prediction_records)
        summary.update(summarize_creativity_extremes(prediction_records))
        summary.update(
            {
                "run_dir": args.run_dir,
                "adapter_dir": str(adapter_dir),
                "eval_file": args.eval_file,
                "train_reference_file": train_reference_file_arg,
                "num_samples": args.num_samples,
                "alpha": args.alpha,
                "lambda_penalty": args.lambda_penalty,
                "novelty_method": args.novelty_method,
                "novelty_judge_model": args.novelty_judge_model or os.environ.get("VERTEX_NOVELTY_JUDGE_MODEL"),
                "embedding_backend": args.embedding_backend,
                "embedding_model_path": args.embedding_model_path,
                "sentence_embedding_model": args.sentence_embedding_model,
                "eval_bf16": run_config.bf16,
                "eval_fp16": run_config.fp16,
                "eval_bnb_4bit_compute_dtype": run_config.bnb_4bit_compute_dtype,
                "output_dir": args.output_dir or str(output_dir),
            }
        )

        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in prediction_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        upload_output_artifacts(output_dir, output_uri, gcs_client)
        log_progress("Evaluation complete")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
