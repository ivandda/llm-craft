"""Offline smoke tests for the LLM-judge creativity scorer.

No network: a StubJudge returns canned plausibility verdicts so the bucketing and
aggregation math is verified deterministically.
"""

from __future__ import annotations

from typing import Sequence

from src.eval.judge_creativity import (
    BATCH_ID_MARKER,
    PrecomputedJudge,
    Verdict,
    aggregate,
    bucket_of,
    build_batch_request_row,
    build_batch_requests,
    is_in_dataset,
    judge_record,
    parse_batch_outputs,
    parse_plausibility_response,
    score_records,
)


class StubJudge:
    """Returns a fixed plausible verdict per exact output string."""

    def __init__(self, plausible_map: dict[str, bool]) -> None:
        self.plausible_map = plausible_map
        self.calls = 0

    def score_recipe(self, input_a: str, input_b: str, outputs: Sequence[str]) -> list[Verdict]:
        self.calls += 1
        return [
            Verdict(plausible=self.plausible_map.get(o, False), plausibility=1.0 if self.plausible_map.get(o, False) else 0.0, reason="stub")
            for o in outputs
        ]


def test_bucket_of():
    assert bucket_of(plausible=False, in_dataset=False) == "invalid"
    assert bucket_of(plausible=False, in_dataset=True) == "invalid"
    assert bucket_of(plausible=True, in_dataset=True) == "valid_known"
    assert bucket_of(plausible=True, in_dataset=False) == "valid_novel"


def test_membership_exact_only():
    known = ["ice cream"]
    known_norm = {"ice cream"}
    assert is_in_dataset("Ice  Cream", known, known_norm, embedder=None, threshold=0.75) is True
    assert is_in_dataset("frozen cream", known, known_norm, embedder=None, threshold=0.75) is False
    assert is_in_dataset("", known, known_norm, embedder=None, threshold=0.75) is False


def test_judge_record_buckets_top1_and_anyk():
    record = {
        "pair_id": "cream+ice",
        "input_a": "cream",
        "input_b": "ice",
        "canonical_output": "ice cream",
        "known_outputs": ["ice cream"],
        # top-1 is a plausible OFF-dataset discovery; one sample matches known.
        "sampled_outputs": ["frozen cream", "ice cream", "frozen cream", "asdfqwer"],
    }
    judge = StubJudge({"frozen cream": True, "ice cream": True, "asdfqwer": False})
    out = judge_record(record, judge, embedder=None, threshold=0.75)

    # Distinct outputs judged once (3 distinct non-empty): frozen cream, ice cream, asdfqwer
    assert judge.calls == 1
    # top-1 "frozen cream": plausible + off-dataset -> valid_novel (a discovery)
    assert out["samples"][0]["bucket"] == "valid_novel"
    assert out["valid_top1"] is True
    assert out["creative_top1"] is True
    # "ice cream" matches known -> valid_known
    assert out["samples"][1]["bucket"] == "valid_known"
    # "asdfqwer" implausible -> invalid
    assert out["samples"][3]["bucket"] == "invalid"
    assert out["valid_anyk"] is True
    assert out["creative_anyk"] is True


def test_judge_record_known_only_is_valid_not_creative():
    record = {
        "input_a": "deer",
        "input_b": "rabbit",
        "canonical_output": "jackalope",
        "known_outputs": ["jackalope"],
        "sampled_outputs": ["jackalope", "jackalope"],
    }
    judge = StubJudge({"jackalope": True})
    out = judge_record(record, judge, embedder=None, threshold=0.75)
    assert out["valid_top1"] is True
    assert out["creative_top1"] is False  # correct but known -> not a discovery
    assert out["creative_anyk"] is False


def test_aggregate_rates():
    judged = [
        {"samples": [{"bucket": "valid_novel"}], "valid_top1": True, "creative_top1": True, "valid_anyk": True, "creative_anyk": True},
        {"samples": [{"bucket": "valid_known"}], "valid_top1": True, "creative_top1": False, "valid_anyk": True, "creative_anyk": False},
        {"samples": [{"bucket": "invalid"}], "valid_top1": False, "creative_top1": False, "valid_anyk": False, "creative_anyk": False},
        {"samples": [{"bucket": "invalid"}], "valid_top1": False, "creative_top1": False, "valid_anyk": True, "creative_anyk": True},
    ]
    scores = aggregate(judged)
    assert scores["n"] == 4
    assert scores["validity_top1"] == 0.5  # 2 of 4 plausible top-1
    assert scores["creativity_top1"] == 0.25  # 1 of 4 discoveries top-1
    assert scores["validity_anyk"] == 0.75
    assert scores["creativity_anyk"] == 0.5
    assert scores["top1_valid_novel_rate"] == 0.25
    assert scores["top1_invalid_rate"] == 0.5


def test_score_records_end_to_end_with_stub():
    records = [
        {"input_a": "a", "input_b": "b", "known_outputs": ["x"], "sampled_outputs": ["x", "y"]},
        {"input_a": "c", "input_b": "d", "known_outputs": ["z"], "sampled_outputs": ["garbage", "garbage"]},
    ]
    judge = StubJudge({"x": True, "y": True, "garbage": False})
    scores, judged = score_records(records, judge, embedder=None, threshold=0.75, concurrency=1)
    assert len(judged) == 2
    # recipe 1 top-1 "x" is known+plausible -> valid, not creative; recipe 2 invalid.
    assert scores["validity_top1"] == 0.5
    assert scores["creativity_top1"] == 0.0
    # recipe 1 any@k has "y" plausible off-dataset -> creative any@k
    assert scores["creativity_anyk"] == 0.5


def test_parse_plausibility_response_by_position():
    text = '{"results": [{"output": "steam", "plausible": true, "plausibility": 0.9, "reason": "ok"}, {"output": "banana", "plausible": false, "plausibility": 0.1, "reason": "off"}]}'
    verdicts = parse_plausibility_response(text, ["steam", "banana"])
    assert verdicts[0].plausible is True and verdicts[0].plausibility == 0.9
    assert verdicts[1].plausible is False


def test_parse_plausibility_truncates_extra_results():
    text = '{"results": [{"output": "a", "plausible": true}, {"output": "b", "plausible": true}, {"output": "c", "plausible": false}]}'
    verdicts = parse_plausibility_response(text, ["a", "b"])
    assert len(verdicts) == 2


# --- batch mode (offline) --- #
def test_build_batch_request_row_embeds_key_and_system_prompt():
    row = build_batch_request_row("fire", "water", ["steam", "mist"])
    text = row["request"]["contents"][0]["parts"][0]["text"]
    assert BATCH_ID_MARKER in text
    assert "fire|water" in text  # normalized recipe key
    assert "steam" in text and "mist" in text
    assert row["request"]["generationConfig"]["responseMimeType"] == "application/json"


def test_build_batch_requests_dedupes_and_indexes():
    records = [
        {"input_a": "fire", "input_b": "water", "sampled_outputs": ["steam", "steam", "mist"]},
        {"input_a": "a", "input_b": "b", "sampled_outputs": ["", ""]},  # all empty -> skipped
    ]
    rows, index = build_batch_requests(records)
    assert len(rows) == 1
    assert index["fire|water"] == ["steam", "mist"]  # deduped, order preserved


def test_parse_batch_outputs_matches_by_key_and_roundtrips_verdicts():
    _, index = build_batch_requests([{"input_a": "fire", "input_b": "water", "sampled_outputs": ["steam", "mist"]}])
    # Simulate a Vertex batch output row: echoed request + model response.
    request_text = f"{BATCH_ID_MARKER} fire|water\n{{...}}"
    response_json = '{"results": [{"output": "steam", "plausible": true, "plausibility": 0.95}, {"output": "mist", "plausible": false, "plausibility": 0.2}]}'
    output_row = {
        "request": {"contents": [{"parts": [{"text": request_text}]}]},
        "response": {"candidates": [{"content": {"parts": [{"text": response_json}]}}]},
    }
    verdicts_by_key = parse_batch_outputs([output_row], index)
    assert "fire|water" in verdicts_by_key
    assert verdicts_by_key["fire|water"]["steam"].plausible is True
    assert verdicts_by_key["fire|water"]["mist"].plausible is False


def test_precomputed_judge_used_in_scoring():
    records = [{"input_a": "fire", "input_b": "water", "known_outputs": ["steam"], "sampled_outputs": ["steam", "mist"]}]
    _, index = build_batch_requests(records)
    request_text = f"{BATCH_ID_MARKER} fire|water\nx"
    response_json = '{"results": [{"output": "steam", "plausible": true}, {"output": "mist", "plausible": true}]}'
    rows = [{
        "request": {"contents": [{"parts": [{"text": request_text}]}]},
        "response": {"candidates": [{"content": {"parts": [{"text": response_json}]}}]},
    }]
    judge = PrecomputedJudge(parse_batch_outputs(rows, index))
    scores, _ = score_records(records, judge, embedder=None, threshold=0.75, concurrency=1)
    # top-1 "steam" is known+plausible -> valid, not creative; "mist" plausible off-dataset -> creative any@k
    assert scores["validity_top1"] == 1.0
    assert scores["creativity_top1"] == 0.0
    assert scores["creativity_anyk"] == 1.0


def test_precomputed_judge_missing_key_defaults_invalid():
    judge = PrecomputedJudge({})
    verdicts = judge.score_recipe("x", "y", ["anything"])
    assert verdicts[0].plausible is False


def test_normalize_batch_state_handles_enum_and_string():
    from src.eval.judge_creativity import _normalize_batch_state

    class EnumLike:
        name = "JOB_STATE_SUCCEEDED"

    class JobEnum:
        state = EnumLike()

    class JobStr:
        state = "JobState.JOB_STATE_RUNNING"

    class JobBare:
        state = "JOB_STATE_FAILED"

    assert _normalize_batch_state(JobEnum()) == "JOB_STATE_SUCCEEDED"
    assert _normalize_batch_state(JobStr()) == "JOB_STATE_RUNNING"
    assert _normalize_batch_state(JobBare()) == "JOB_STATE_FAILED"
