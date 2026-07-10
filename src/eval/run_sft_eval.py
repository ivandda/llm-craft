from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel

from src.eval.creativity import (
    CreativityComponents,
    compute_creativity_components,
    normalized_mean_cosine_distances,
)
from src.eval.embeddings import BaseTextEmbedder, build_text_embedder
from src.eval.vertex_judge import load_vertex_environment
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
    parser.add_argument("--output_dir", default=None, help="Directory to write predictions and summary.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--max_concept_words", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--lambda_penalty", type=float, default=2.0)
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
    parser.add_argument(
        "--embedding_device",
        default="cpu",
        help="Device for the embedding backend. Recommended: cpu for sentence embeddings on Vertex eval jobs.",
    )
    parser.add_argument("--eval_bf16", type=_str_to_bool, default=None)
    parser.add_argument("--eval_fp16", type=_str_to_bool, default=None)
    parser.add_argument("--eval_bnb_4bit_compute_dtype", default=None)
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument(
        "--no_adapter",
        action="store_true",
        help="Evaluate the base model without a LoRA adapter (baseline). Still reads run_dir/config.yaml for the generation setup.",
    )
    parser.add_argument(
        "--enable_thinking",
        type=_str_to_bool,
        default=None,
        help="qwen_chat only: value passed to the chat template's enable_thinking. Default None keeps the template default.",
    )
    parser.add_argument(
        "--strip_think",
        type=_str_to_bool,
        default=False,
        help="Drop any <think>...</think> reasoning block from generations before extracting the concept.",
    )
    parser.add_argument(
        "--close_think_prompt",
        type=_str_to_bool,
        default=False,
        help="Pre-close an empty <think></think> in the prompt so a thinking base model answers directly.",
    )
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


def download_gcs_prefix(
    uri: str, local_dir: Path, client: Any, exclude_dirs: list[str] | None = None
) -> Path:
    log_progress(f"Downloading directory from {uri} to {local_dir}")
    bucket_name, blob_prefix = parse_gcs_uri(uri)
    prefix = blob_prefix.rstrip("/")
    if not prefix:
        raise ValueError(f"GCS directory URI must include a prefix: {uri!r}")
    excluded = set(exclude_dirs or [])
    bucket = client.bucket(bucket_name)
    found_blob = False
    for blob in client.list_blobs(bucket, prefix=prefix + "/"):
        if blob.name.endswith("/"):
            continue
        relative_path = Path(blob.name).relative_to(prefix)
        # Skip whole subdirectories (e.g. multi-GB training checkpoints the eval never uses).
        if relative_path.parts and relative_path.parts[0] in excluded:
            continue
        found_blob = True
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
    # The eval only needs config.yaml + the adapter; checkpoints can be 100+ GB and
    # would blow the worker disk and startup time, so skip them.
    return download_gcs_prefix(
        run_dir, staging_root / "run_dir", client, exclude_dirs=["checkpoints"]
    )


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


def render_generation_prompt(
    record: EvalRecord,
    config: SFTConfig,
    tokenizer: Any,
    *,
    enable_thinking: bool | None = None,
    close_think: bool = False,
) -> str:
    if config.prompt_format == "qwen_chat":
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": render_user_prompt(record.input_a, record.input_b)})
        template_kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        if close_think and "<think>" in prompt and "</think>" not in prompt:
            # Pre-close an empty reasoning block so a thinking model answers directly
            # instead of spending the whole budget inside <think>.
            prompt = prompt.rstrip() + "\n</think>\n\n"
        return prompt
    return render_prefix(record.input_a, record.input_b)


def strip_think_block(text: str) -> str:
    close = "</think>"
    if close in text:
        return text.split(close, 1)[-1]
    if "<think>" in text:
        # Reasoning was opened but never closed (e.g. truncated by max_new_tokens):
        # there is no usable final answer.
        return ""
    return text


def decode_generated_text(
    tokenizer: Any,
    sequences: torch.Tensor,
    prompt_length: int,
) -> list[str]:
    decoded = tokenizer.batch_decode(sequences[:, prompt_length:], skip_special_tokens=True)
    return [text.strip() for text in decoded]


def postprocess_generated_concept(text: str, *, max_words: int = 3) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.splitlines()[0].strip()
    cleaned = re.split(r"[.:;?!\(\)\[\]\{\}]", cleaned, maxsplit=1)[0].strip()
    cleaned = cleaned.replace("*", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    connector_match = re.search(r"\b(with|and|or|of|in|for|to|on|from|by)\b", cleaned, flags=re.IGNORECASE)
    if connector_match is not None:
        cleaned = cleaned[: connector_match.start()].strip()
    cleaned = cleaned.strip(" ,-_")
    words = [word.strip(" ,-_") for word in cleaned.split() if word.strip(" ,-_")]
    if max_words > 0:
        words = words[:max_words]
    return " ".join(words)


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
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        log_progress(
            "Skipping explicit model.to(...) because the model is quantized and already managed by bitsandbytes/device_map"
        )
        return model
    if getattr(model, "hf_device_map", None) is not None:
        log_progress("Skipping explicit model.to(...) because the model already has an hf_device_map")
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


def build_eval_embedder(args: argparse.Namespace) -> BaseTextEmbedder:
    return build_text_embedder(
        args.embedding_backend,
        word_vector_path=args.embedding_model_path,
        sentence_transformer_model=args.sentence_embedding_model,
        device=args.embedding_device,
    )


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path.cwd()
    load_vertex_environment(repo_root)
    log_progress("Starting creativity evaluation pipeline")
    gcs_client = build_gcs_client() if any(
        is_gcs_uri(value)
        for value in [args.run_dir, args.eval_file, args.output_dir or ""]
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
        if args.no_adapter:
            adapter_dir = None
            log_progress("Base-model baseline: skipping adapter (--no_adapter)")
        else:
            adapter_dir = resolve_adapter_dir(run_dir, args.adapter_dir)
            log_progress(f"Using adapter directory {adapter_dir}")
        eval_file = stage_input_file(args.eval_file, staging_root, gcs_client, "eval_file")
        output_dir, output_uri = prepare_output_dir(args.output_dir, run_dir, str(eval_file), staging_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_progress(f"Resolved eval file to {eval_file}")
        log_progress(f"Writing outputs to {output_uri or output_dir}")

        log_progress(f"Loading base model and tokenizer from {run_config.model_name_or_path}")
        model, tokenizer = build_model_and_tokenizer(run_config, apply_lora=False)
        if adapter_dir is not None:
            log_progress("Loading PEFT adapter")
            model = PeftModel.from_pretrained(model, adapter_dir)
        model = maybe_move_to_device(model, args.device)
        model.eval()
        log_progress(f"Model ready on device {next(model.parameters()).device}")

        eval_records = load_eval_records(eval_file, max_examples=args.max_examples)
        log_progress(f"Loaded {len(eval_records)} eval recipes")
        log_progress(
            "Building embedder "
            f"with backend={args.embedding_backend} model={args.sentence_embedding_model} "
            f"device={args.embedding_device}"
        )
        embedder = build_eval_embedder(args)

        prediction_records: list[dict[str, Any]] = []
        log_progress(f"Starting generation for {len(eval_records)} recipes with num_samples={args.num_samples}")
        for index, record in enumerate(eval_records, start=1):
            prompt = render_generation_prompt(
                record,
                run_config,
                tokenizer,
                enable_thinking=args.enable_thinking,
                close_think=args.close_think_prompt,
            )
            encoded = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
            generation = model.generate(
                **encoded,
                do_sample=args.num_samples > 1,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            sampled_outputs = decode_generated_text(tokenizer, generation, encoded["input_ids"].shape[1])
            if args.strip_think:
                sampled_outputs = [strip_think_block(output) for output in sampled_outputs]
            sampled_outputs = [
                postprocess_generated_concept(output, max_words=args.max_concept_words) for output in sampled_outputs
            ]
            if not sampled_outputs:
                sampled_outputs = [""]

            sample_embeddings = embedder.encode(sampled_outputs)
            recipe_candidate_embeddings = embedder.encode(record.known_outputs)
            input_embeddings = embedder.encode([record.input_a, record.input_b])
            novelty_scores = normalized_mean_cosine_distances(sample_embeddings, input_embeddings)

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
                "adapter_dir": str(adapter_dir) if adapter_dir is not None else "none (base model)",
                "no_adapter": args.no_adapter,
                "enable_thinking": args.enable_thinking,
                "strip_think": args.strip_think,
                "close_think_prompt": args.close_think_prompt,
                "eval_file": args.eval_file,
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "max_concept_words": args.max_concept_words,
                "alpha": args.alpha,
                "lambda_penalty": args.lambda_penalty,
                "novelty_method": "input_distance",
                "novelty_embedding_definition": "mean_cosine_distance_to_input_a_and_input_b_divided_by_2",
                "embedding_backend": args.embedding_backend,
                "embedding_model_path": args.embedding_model_path,
                "sentence_embedding_model": args.sentence_embedding_model,
                "embedding_device": args.embedding_device,
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
