import json
import os

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_SFT_SMOKE") != "1",
    reason="Set RUN_SFT_SMOKE=1 to run the model-download DPO smoke training.",
)
def test_smoke_train_dpo_with_tiny_model(tmp_path):
    """End-to-end DPO plumbing on CPU: SFT 1 step -> init DPO from that adapter ->
    precompute reference -> concatenated forward -> backward -> checkpoint/metrics."""
    from src.sft.train import main

    # 1) Produce a tiny SFT adapter to initialize the DPO policy/reference from.
    recipe = (
        '{"input_a":"fire","input_b":"water","candidate_outputs":'
        '[{"output":"steam","source":"observed","rank":1}]}\n'
    )
    sft_train = tmp_path / "sft_train.jsonl"
    sft_train.write_text(recipe * 2, encoding="utf-8")
    sft_dev = tmp_path / "sft_dev.jsonl"
    sft_dev.write_text(recipe, encoding="utf-8")
    sft_out = tmp_path / "sft_runs"
    common_model = ["--model_name_or_path", "sshleifer/tiny-gpt2",
                    "--load_in_4bit", "false", "--bf16", "false", "--fp16", "false",
                    "--prompt_format", "plain"]
    main(common_model + [
        "--train_path", str(sft_train), "--dev_path", str(sft_dev),
        "--output_dir", str(sft_out), "--run_name", "smoke_sft",
        "--max_steps", "1", "--eval_steps", "1", "--save_steps", "1", "--logging_steps", "1",
    ])
    sft_runs = list(sft_out.glob("*smoke_sft*"))
    assert sft_runs, "SFT smoke run dir not found"
    init_adapter = sft_runs[0] / "final_adapter"
    assert init_adapter.exists()

    # 2) DPO preference pairs (chosen short/valid, rejected verbose).
    pair = json.dumps(
        {"input_a": "fire", "input_b": "water", "chosen": "steam",
         "rejected": "a hot rising cloud of water vapor everywhere"}
    ) + "\n"
    dpo_train = tmp_path / "pairs_train.jsonl"
    dpo_train.write_text(pair * 2, encoding="utf-8")
    dpo_dev = tmp_path / "pairs_dev.jsonl"
    dpo_dev.write_text(pair, encoding="utf-8")
    dpo_out = tmp_path / "dpo_runs"

    # 3) DPO run initialized from the SFT adapter.
    main(common_model + [
        "--objective", "dpo",
        "--init_adapter_path", str(init_adapter),
        "--train_path", str(dpo_train), "--dev_path", str(dpo_dev),
        "--output_dir", str(dpo_out), "--run_name", "smoke_dpo",
        "--dpo_beta", "0.1", "--lora_dropout", "0.0", "--max_seq_length", "64",
        "--max_steps", "2", "--eval_steps", "2", "--save_steps", "2", "--logging_steps", "1",
    ])

    dpo_runs = list(dpo_out.glob("*smoke_dpo*"))
    assert dpo_runs, "DPO smoke run dir not found"
    dpo_run = dpo_runs[0]
    assert (dpo_run / "final_adapter").exists()

    # metrics.jsonl should carry DPO diagnostics (reward_margin/reward_accuracy).
    metrics = [json.loads(line) for line in (dpo_run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train_rows = [m for m in metrics if m.get("split") == "train"]
    assert train_rows, "no train metrics logged"
    assert any("reward_margin" in m for m in train_rows)
