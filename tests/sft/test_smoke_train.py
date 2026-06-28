import os

import pytest


@pytest.mark.skipif(os.environ.get("RUN_SFT_SMOKE") != "1", reason="Set RUN_SFT_SMOKE=1 to run model-download smoke training.")
def test_smoke_train_with_tiny_model(tmp_path):
    from src.sft.train import main

    train_path = tmp_path / "train.jsonl"
    dev_path = tmp_path / "dev.jsonl"
    row = (
        '{"input_a":"fire","input_b":"water","candidate_outputs":'
        '[{"output":"steam","source":"observed","rank":1},{"output":"vapor","source":"teacher","rank":2}]}\n'
    )
    train_path.write_text(row * 2, encoding="utf-8")
    dev_path.write_text(row, encoding="utf-8")

    main(
        [
            "--model_name_or_path",
            "sshleifer/tiny-gpt2",
            "--train_path",
            str(train_path),
            "--dev_path",
            str(dev_path),
            "--output_dir",
            str(tmp_path / "runs"),
            "--run_name",
            "pytest_smoke",
            "--load_in_4bit",
            "false",
            "--bf16",
            "false",
            "--fp16",
            "false",
            "--max_steps",
            "1",
            "--eval_steps",
            "1",
            "--save_steps",
            "1",
            "--logging_steps",
            "1",
        ]
    )
