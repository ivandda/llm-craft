import pytest

from src.data.export_eval import ALL_SIZE, format_size, parse_size, should_keep_record


def test_parse_size_accepts_positive_int_and_all():
    assert parse_size(1000, "dev_keep") == 1000
    assert parse_size("all", "dev_keep") == ALL_SIZE
    assert parse_size(" ALL ", "dev_keep") == ALL_SIZE


def test_parse_size_rejects_typos_and_non_positive_values():
    with pytest.raises(ValueError, match="positive integer or 'all'"):
        parse_size("alll", "test_keep")

    with pytest.raises(ValueError, match="positive integer or 'all'"):
        parse_size(0, "test_keep")


def test_format_size_handles_all():
    assert format_size(1000) == "1k"
    assert format_size(1500) == "1500"
    assert format_size(ALL_SIZE) == "all"


def test_should_keep_record_appends_every_record_when_limit_is_all():
    reservoir = [{"existing": True}]

    assert should_keep_record(reservoir, ALL_SIZE, seen_count=2) == 1
