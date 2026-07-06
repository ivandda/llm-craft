from src.eval.score_coverage import score_records


def _record(sampled_outputs, known_outputs, canonical_output):
    return {
        "sampled_outputs": sampled_outputs,
        "known_outputs": known_outputs,
        "canonical_output": canonical_output,
    }


def test_score_records_top1_and_anyk_coverage():
    records = [
        # top-1 hits a known output (and the canonical)
        _record(["steam", "mist"], ["steam", "mist"], "steam"),
        # top-1 misses, but a later sample hits a known output -> only any@k
        _record(["vapor cloud", "mist"], ["steam", "mist"], "steam"),
        # nothing matches
        _record(["banana", "rock"], ["steam", "mist"], "steam"),
    ]

    scores = score_records(records)

    assert scores["n"] == 3
    assert scores["top1_known_match"] == 1 / 3
    assert scores["anyk_known_match"] == 2 / 3
    assert scores["top1_canonical_match"] == 1 / 3
    assert scores["anyk_canonical_match"] == 1 / 3


def test_score_records_normalizes_case_and_whitespace():
    records = [_record(["  Ice   Cream "], ["ice cream"], "ice cream")]

    scores = score_records(records)

    assert scores["top1_known_match"] == 1.0


def test_score_records_tracks_verbosity_and_empty():
    records = [
        _record(["ice cream"], ["ice cream"], "ice cream"),  # 2 words
        _record(["robot vacuum cleaner fire"], ["black hole"], "black hole"),  # 4 words
        _record([""], ["kelp"], "kelp"),  # empty
    ]

    scores = score_records(records)

    assert scores["empty_top1_rate"] == 1 / 3
    assert scores["frac_top1_le2_words"] == 1 / 3
    assert abs(scores["mean_top1_words"] - (2 + 4 + 0) / 3) < 1e-9


def test_score_records_falls_back_to_prediction_when_no_samples():
    records = [{"prediction": "steam", "known_outputs": ["steam"], "canonical_output": "steam"}]

    scores = score_records(records)

    assert scores["top1_known_match"] == 1.0


def test_score_records_empty_input():
    assert score_records([])["n"] == 0
