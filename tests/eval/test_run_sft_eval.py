import sys
from types import SimpleNamespace

from src.eval.run_sft_eval import (
    apply_eval_precision_overrides,
    build_output_record,
    is_gcs_uri,
    maybe_move_to_device,
    parse_args,
    parse_gcs_uri,
    postprocess_generated_concept,
    prepare_output_dir,
    strip_think_block,
    summarize_creativity_extremes,
)
from src.sft.config import SFTConfig


def test_build_output_record_keeps_only_creativity_context_fields():
    eval_record = {
        "pair_id": "pair-1",
        "input_a": "fire",
        "input_b": "water",
        "canonical_output": "steam",
        "known_outputs": ["steam", "mist"],
    }

    output_record = build_output_record(eval_record, "Mist")

    assert output_record == {
        "pair_id": "pair-1",
        "input_a": "fire",
        "input_b": "water",
        "prediction": "Mist",
        "canonical_output": "steam",
        "known_outputs": ["steam", "mist"],
    }


def test_parse_args_defaults_to_dev_eval_set(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_sft_eval", "--run_dir", "runs/sft/example"])

    args = parse_args()

    assert args.eval_file == "datasets/processed/eval_dev_all.jsonl"
    assert args.embedding_device == "cpu"
    assert args.repetition_penalty == 1.15
    assert args.no_repeat_ngram_size == 3
    assert args.max_concept_words == 3


def test_apply_eval_precision_overrides_updates_run_config_for_t4(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sft_eval",
            "--run_dir",
            "runs/sft/example",
            "--eval_bf16",
            "false",
            "--eval_fp16",
            "true",
            "--eval_bnb_4bit_compute_dtype",
            "float16",
        ],
    )

    args = parse_args()
    config = SFTConfig(bf16=True, fp16=False, bnb_4bit_compute_dtype="bfloat16")
    updated = apply_eval_precision_overrides(config, args)

    assert updated.bf16 is False
    assert updated.fp16 is True
    assert updated.bnb_4bit_compute_dtype == "float16"


def test_parse_gcs_uri_extracts_bucket_and_blob_prefix():
    assert is_gcs_uri("gs://llm-craft-bucket/runs/model-a") is True
    assert parse_gcs_uri("gs://llm-craft-bucket/runs/model-a") == (
        "llm-craft-bucket",
        "runs/model-a",
    )


def test_prepare_output_dir_uses_local_staging_for_gcs_output(tmp_path):
    output_dir, output_uri = prepare_output_dir(
        "gs://llm-craft-bucket/eval_outputs/run-1",
        tmp_path / "run_dir",
        "test.jsonl",
        tmp_path,
    )

    assert output_dir == tmp_path / "output"
    assert output_uri == "gs://llm-craft-bucket/eval_outputs/run-1"


def test_maybe_move_to_device_skips_explicit_move_for_4bit_models():
    model = SimpleNamespace(is_loaded_in_4bit=True)

    result = maybe_move_to_device(model, "cuda")

    assert result is model


def test_postprocess_generated_concept_keeps_first_compact_span():
    output = "poisoned animal model for lead poisoning animal model.\nRationale: because..."

    result = postprocess_generated_concept(output)

    assert result == "poisoned animal model"


def test_postprocess_generated_concept_stops_before_connector_phrase():
    output = "clockwork watch with clockwork timer or"

    result = postprocess_generated_concept(output)

    assert result == "clockwork watch"


def test_parse_args_base_baseline_flags_default_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_sft_eval", "--run_dir", "runs/sft/example"])

    args = parse_args()

    assert args.no_adapter is False
    assert args.enable_thinking is None
    assert args.strip_think is False


def test_parse_args_accepts_base_baseline_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sft_eval",
            "--run_dir",
            "runs/sft/example",
            "--no_adapter",
            "--enable_thinking",
            "false",
            "--strip_think",
            "true",
        ],
    )

    args = parse_args()

    assert args.no_adapter is True
    assert args.enable_thinking is False
    assert args.strip_think is True


def test_strip_think_block_removes_closed_reasoning():
    assert strip_think_block("<think>\nreasoning here\n</think>\nsteam").strip() == "steam"


def test_strip_think_block_drops_unclosed_reasoning():
    # Truncated by max_new_tokens: opened <think> but never reached the answer.
    assert strip_think_block("<think>\nlong reasoning that never finished") == ""


def test_strip_think_block_passes_through_plain_text():
    assert strip_think_block("ice cream") == "ice cream"


def test_summarize_creativity_extremes_includes_min_and_max_records():
    records = [
        {
            "pair_id": "pair-low",
            "input_a": "fire",
            "input_b": "stone",
            "prediction": "ash",
            "canonical_output": "lava",
            "known_outputs": ["lava"],
            "sampled_outputs": ["ash", "dust"],
            "creativity": {
                "plausibility_distance": 1.0,
                "plausibility_score": 0.5,
                "novelty": 0.2,
                "diversity_distance": 0.4,
                "diversity_score": 0.8,
                "local_creativity": 0.18,
            },
        },
        {
            "pair_id": "pair-high",
            "input_a": "water",
            "input_b": "wind",
            "prediction": "storm",
            "canonical_output": "storm",
            "known_outputs": ["storm", "mist"],
            "sampled_outputs": ["storm", "tempest"],
            "creativity": {
                "plausibility_distance": 0.2,
                "plausibility_score": 0.9,
                "novelty": 0.8,
                "diversity_distance": 0.3,
                "diversity_score": 0.85,
                "local_creativity": 0.74,
            },
        },
    ]

    summary = summarize_creativity_extremes(records)

    assert summary["min_local_creativity_recipe"]["pair_id"] == "pair-low"
    assert summary["min_local_creativity_recipe"]["sampled_outputs"] == ["ash", "dust"]
    assert summary["max_local_creativity_recipe"]["pair_id"] == "pair-high"
    assert summary["max_local_creativity_recipe"]["sampled_outputs"] == ["storm", "tempest"]
