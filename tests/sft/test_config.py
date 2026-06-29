import pytest

from src.sft.config import config_from_args, parse_args


def test_optional_example_limits_parse_as_ints():
    args = parse_args(["--max_train_examples", "32", "--max_dev_examples", "16"])

    config = config_from_args(args)

    assert config.max_train_examples == 32
    assert config.max_dev_examples == 16


def test_loss_type_alias_populates_candidate_loss_axes():
    args = parse_args(["--loss_type", "ce"])

    config = config_from_args(args)

    assert config.candidate_weighting == "uniform"
    assert config.candidate_aggregation == "expected_logprob"


def test_explicit_candidate_loss_axes_override_loss_alias(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "loss_type: ce",
                "candidate_weighting: dataset",
                "candidate_aggregation: logsumexp_prob",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    config = config_from_args(args)

    assert config.candidate_weighting == "dataset"
    assert config.candidate_aggregation == "logsumexp_prob"


def test_partial_explicit_candidate_loss_axes_are_rejected_from_cli():
    with pytest.raises(ValueError, match="you must set both"):
        config_from_args(parse_args(["--loss_type", "ce", "--candidate_weighting", "dataset"]))


def test_partial_explicit_candidate_loss_axes_are_rejected_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "loss_type: ce",
                "candidate_weighting: dataset",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="you must set both"):
        config_from_args(parse_args(["--config", str(config_path)]))


def test_concept_set_uniform_alias_populates_both_axes():
    args = parse_args(["--loss_type", "concept_set_uniform"])

    config = config_from_args(args)

    assert config.candidate_weighting == "uniform"
    assert config.candidate_aggregation == "logsumexp_prob"


def test_validate_config_rejects_invalid_candidate_weighting():
    with pytest.raises(ValueError, match="candidate_weighting"):
        config_from_args(parse_args(["--candidate_weighting", "bad"]))


def test_prompt_format_accepts_qwen_chat():
    args = parse_args(["--prompt_format", "qwen_chat"])

    config = config_from_args(args)

    assert config.prompt_format == "qwen_chat"


def test_validate_config_rejects_invalid_prompt_format():
    with pytest.raises(ValueError, match="prompt_format"):
        config_from_args(parse_args(["--prompt_format", "bad"]))
