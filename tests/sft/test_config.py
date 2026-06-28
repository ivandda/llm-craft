from src.sft.config import config_from_args, parse_args


def test_optional_example_limits_parse_as_ints():
    args = parse_args(["--max_train_examples", "32", "--max_dev_examples", "16"])

    config = config_from_args(args)

    assert config.max_train_examples == 32
    assert config.max_dev_examples == 16
