import json

import pytest

from src.data.enrich_rationales import (
    RationaleConfig,
    TeacherResponseError,
    enrich_record_with_rationales,
    enrich_split,
    parse_teacher_rationales,
    parse_args,
)


class FakeRationaleTeacher:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, input_a, input_b, candidates, max_rationale_words):
        self.calls.append(
            {
                "input_a": input_a,
                "input_b": input_b,
                "candidates": [dict(candidate) for candidate in candidates],
                "max_rationale_words": max_rationale_words,
            }
        )
        if not self.responses:
            return json.dumps({"rationales": []})
        return self.responses.pop(0)


def source_record():
    return {
        "input_a": "fire",
        "input_b": "water",
        "candidate_outputs": [
            {"output": "steam", "source": "observed"},
            {"output": "sauna", "source": "teacher"},
        ],
        "quality_status": "complete",
        "metadata": {
            "source_dataset": "dataset_01_teacher_enriched_multi_output_no_rationale",
            "source_split": "train",
            "teacher_provider": "google_vertex_ai",
            "teacher_model": "gemini-2.5-flash",
            "enrichment_version": "teacher_multi_output_no_rationale_v1",
            "target_num_outputs": 5,
            "has_rationales": False,
        },
    }


def test_enrich_record_adds_rationale_to_each_candidate_without_changing_outputs():
    teacher = FakeRationaleTeacher(
        [
            json.dumps(
                {
                    "rationales": [
                        {
                            "output": "steam",
                            "rationale": "Fire heats water until it turns into steam.",
                        },
                        {
                            "output": "sauna",
                            "rationale": "Heat and water create the steamy setting of a sauna.",
                        },
                    ]
                }
            )
        ]
    )

    record, failure = enrich_record_with_rationales(
        source_record(),
        split="train",
        teacher=teacher,
        config=RationaleConfig(max_retries=0),
    )

    assert failure is None
    assert [candidate["output"] for candidate in record["candidate_outputs"]] == ["steam", "sauna"]
    assert [candidate["source"] for candidate in record["candidate_outputs"]] == ["observed", "teacher"]
    assert record["candidate_outputs"][0]["rationale"] == "Fire heats water until it turns into steam."
    assert record["candidate_outputs"][1]["rationale"] == "Heat and water create the steamy setting of a sauna."
    assert record["quality_status"] == "complete"
    assert record["metadata"]["source_dataset"] == "dataset_01_teacher_enriched_multi_output_no_rationale"
    assert record["metadata"]["enrichment_version"] == "teacher_multi_output_with_rationale_v1"
    assert record["metadata"]["has_rationales"] is True
    assert record["metadata"]["rationale_language"] == "en"


def test_parse_teacher_rationales_rejects_invalid_json():
    with pytest.raises(TeacherResponseError):
        parse_teacher_rationales("steam: because fire heats water", ["steam"])


def test_parse_teacher_rationales_rejects_changed_outputs():
    with pytest.raises(TeacherResponseError):
        parse_teacher_rationales(
            json.dumps({"rationales": [{"output": "mist", "rationale": "Water becomes mist."}]}),
            ["steam"],
        )


def test_enrich_record_marks_partial_when_a_rationale_is_missing():
    teacher = FakeRationaleTeacher(
        [
            json.dumps(
                {
                    "rationales": [
                        {
                            "output": "steam",
                            "rationale": "Fire heats water until it turns into steam.",
                        }
                    ]
                }
            )
        ]
    )

    record, failure = enrich_record_with_rationales(
        source_record(),
        split="dev",
        teacher=teacher,
        config=RationaleConfig(max_retries=0),
    )

    assert record["quality_status"] == "partial_rationale"
    assert record["candidate_outputs"][0]["rationale"] == "Fire heats water until it turns into steam."
    assert "rationale" not in record["candidate_outputs"][1]
    assert failure is not None
    assert failure["reason"] == "missing_rationales"
    assert failure["missing_outputs"] == ["sauna"]


def test_enrich_record_marks_failed_when_teacher_response_is_invalid():
    teacher = FakeRationaleTeacher(["not json"])

    record, failure = enrich_record_with_rationales(
        source_record(),
        split="test",
        teacher=teacher,
        config=RationaleConfig(max_retries=0),
    )

    assert record["quality_status"] == "failed_rationale"
    assert all("rationale" not in candidate for candidate in record["candidate_outputs"])
    assert failure is not None
    assert failure["reason"] == "teacher_response_error"


def test_enrich_split_resume_skips_records_that_already_have_all_rationales(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    failed_path = tmp_path / "failed.jsonl"
    complete_record = source_record()
    for candidate in complete_record["candidate_outputs"]:
        candidate["rationale"] = f"{candidate['output']} is explained."
    complete_record["metadata"]["has_rationales"] = True

    input_path.write_text(json.dumps(source_record()) + "\n", encoding="utf-8")
    output_path.write_text(json.dumps(complete_record) + "\n", encoding="utf-8")
    teacher = FakeRationaleTeacher([])

    counts = enrich_split(
        input_path=input_path,
        output_path=output_path,
        failed_path=failed_path,
        split="train",
        teacher=teacher,
        config=RationaleConfig(),
        limit=None,
        resume=True,
    )

    assert teacher.calls == []
    assert counts["skipped_existing"] == 1
    assert counts["written"] == 0
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1


def test_parse_args_rejects_dry_run(monkeypatch):
    monkeypatch.setattr("sys.argv", ["enrich_rationales", "--dry-run"])

    with pytest.raises(SystemExit):
        parse_args()
