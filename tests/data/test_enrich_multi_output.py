import json

import pytest

from src.data.enrich_multi_output import (
    EnrichmentConfig,
    TeacherResponseError,
    build_enriched_record,
    enrich_recipe,
    parse_teacher_outputs,
)


class FakeTeacher:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, input_a, input_b, existing_outputs, num_outputs):
        self.calls.append(
            {
                "input_a": input_a,
                "input_b": input_b,
                "existing_outputs": list(existing_outputs),
                "num_outputs": num_outputs,
            }
        )
        if not self.responses:
            return json.dumps({"outputs": []})
        return self.responses.pop(0)


def test_build_enriched_record_preserves_observed_outputs_and_completes_with_teacher_outputs():
    record = build_enriched_record(
        recipe={"input_a": "fire", "input_b": "water", "outputs": ["Steam", "mist"]},
        teacher_outputs=["vapor", "hot spring", "sauna"],
        split="train",
        config=EnrichmentConfig(),
    )

    assert record["input_a"] == "fire"
    assert record["input_b"] == "water"
    assert record["quality_status"] == "complete"
    assert record["candidate_outputs"] == [
        {"output": "steam", "source": "observed"},
        {"output": "mist", "source": "observed"},
        {"output": "vapor", "source": "teacher"},
        {"output": "hot spring", "source": "teacher"},
        {"output": "sauna", "source": "teacher"},
    ]


def test_enrich_recipe_requests_only_missing_outputs_from_teacher():
    teacher = FakeTeacher([json.dumps({"outputs": ["vapor", "hot spring"]})])

    record, failure = enrich_recipe(
        recipe={"input_a": "fire", "input_b": "water", "outputs": ["steam", "mist", "sauna"]},
        split="dev",
        teacher=teacher,
        config=EnrichmentConfig(max_retries=0),
    )

    assert failure is None
    assert teacher.calls == [
        {
            "input_a": "fire",
            "input_b": "water",
            "existing_outputs": ["steam", "mist", "sauna"],
            "num_outputs": 2,
        }
    ]
    assert [candidate["output"] for candidate in record["candidate_outputs"]] == [
        "steam",
        "mist",
        "sauna",
        "vapor",
        "hot spring",
    ]


def test_enrich_recipe_does_not_call_teacher_when_observed_outputs_are_enough():
    teacher = FakeTeacher([])

    record, failure = enrich_recipe(
        recipe={
            "input_a": "fire",
            "input_b": "water",
            "outputs": ["steam", "mist", "vapor", "hot spring", "sauna", "rain"],
        },
        split="test",
        teacher=teacher,
        config=EnrichmentConfig(),
    )

    assert failure is None
    assert teacher.calls == []
    assert record["quality_status"] == "complete"
    assert [candidate["output"] for candidate in record["candidate_outputs"]] == [
        "steam",
        "mist",
        "vapor",
        "hot spring",
        "sauna",
    ]


def test_enrich_recipe_discards_duplicate_teacher_outputs_and_marks_partial():
    teacher = FakeTeacher([json.dumps({"outputs": ["steam", "vapor", "vapor", "fire"]})])

    record, failure = enrich_recipe(
        recipe={"input_a": "fire", "input_b": "water", "outputs": ["steam"]},
        split="train",
        teacher=teacher,
        config=EnrichmentConfig(max_retries=0),
    )

    assert failure is not None
    assert record["quality_status"] == "partial_enrichment"
    assert [candidate["output"] for candidate in record["candidate_outputs"]] == ["steam", "vapor"]
    assert failure["reason"] == "insufficient_valid_teacher_outputs"


def test_parse_teacher_outputs_rejects_invalid_json():
    with pytest.raises(TeacherResponseError):
        parse_teacher_outputs("mist, vapor, sauna")
