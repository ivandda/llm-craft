import sys

from src.eval.run_sft_eval import build_output_record, parse_args, summarize_output_records


def test_build_output_record_includes_prediction_and_match_flags():
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
        "exact_canonical_match": False,
        "known_output_match": True,
        "is_empty_prediction": False,
    }


def test_summarize_output_records_counts_accuracy_and_empty_predictions():
    records = [
        {
            "exact_canonical_match": True,
            "known_output_match": True,
            "is_empty_prediction": False,
        },
        {
            "exact_canonical_match": False,
            "known_output_match": True,
            "is_empty_prediction": False,
        },
        {
            "exact_canonical_match": False,
            "known_output_match": False,
            "is_empty_prediction": True,
        },
    ]

    summary = summarize_output_records(records)

    assert summary == {
        "num_examples": 3,
        "canonical_accuracy": 1 / 3,
        "known_output_accuracy": 2 / 3,
        "empty_predictions": 1,
    }


def test_parse_args_defaults_to_dev_eval_set(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_sft_eval"])

    args = parse_args()

    assert args.eval_file == "datasets/processed/eval_dev_all.jsonl"
