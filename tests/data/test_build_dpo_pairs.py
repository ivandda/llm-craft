"""Offline tests for the DPO pair-selection rules (no model)."""

from __future__ import annotations

from src.data.build_dpo_pairs import select_preference_pair


def _select(samples, known=("steam",), canonical="steam", candidate_rows=None):
    return select_preference_pair(
        "fire", "water", list(known), canonical, samples, candidate_rows=candidate_rows
    )


def test_chosen_on_policy_short_valid_rejected_verbose():
    pair = _select(["steam", "steam", "a hot cloud of water vapor", "steam vapor rising slowly"])
    assert pair is not None
    assert pair["chosen"] == "steam"
    assert pair["chosen_source"] == "on_policy"
    assert pair["rejected_len"] >= 3
    assert pair["rejected_source"] == "on_policy_verbose"


def test_rejected_can_be_short_invalid_not_only_verbose():
    # length-bias mitigation: a short-but-wrong sample is a valid negative.
    pair = _select(["steam", "banana"])
    assert pair["chosen"] == "steam"
    assert pair["rejected"] == "banana"
    assert pair["rejected_source"] == "on_policy_invalid"
    assert pair["rejected_valid"] is False


def test_chosen_falls_back_to_dataset_candidate():
    rows = [
        {"output": "steam", "source": "observed", "rank": 1},
        {"output": "water vapor cloud", "source": "teacher", "rank": 2},
    ]
    pair = _select(["hot steam vapor", "a big rising cloud"], candidate_rows=rows)
    assert pair is not None
    assert pair["chosen"] == "steam"
    assert pair["chosen_source"] == "dataset_candidate"
    assert pair["rejected_len"] >= 3


def test_skip_when_all_samples_short_and_valid():
    # nothing to push down -> no pair.
    assert _select(["steam", "steam"]) is None


def test_skip_when_no_short_chosen_available():
    # no on-policy short valid and no short dataset candidate.
    rows = [{"output": "water vapor cloud", "source": "teacher", "rank": 1}]
    assert _select(["a long verbose description", "another long phrase here"], candidate_rows=rows) is None


def test_rejected_differs_from_chosen():
    pair = _select(["steam", "steam vapor mist cloud"])
    assert pair is not None
    assert pair["chosen"] != pair["rejected"]


def test_top1_bad_sample_is_preferred_as_rejected():
    # the model's actual top-1 (index 0) is the verbose failure -> use it as rejected.
    pair = _select(["steam vapor rising fast", "steam", "steam"])
    assert pair["chosen"] == "steam"
    assert pair["rejected"] == "steam vapor rising fast"


def test_empty_samples_are_not_used_as_rejected():
    # empty generations are not useful negatives; require a non-empty verbose/invalid one.
    assert _select(["steam", "", ""]) is None  # only bad candidate is empty -> skip
