from src.eval.metrics import evaluate_prediction, normalize_answer


def test_normalize_answer_lowercases_strips_and_collapses_spaces():
    assert normalize_answer("  Hot   Spring  ") == "hot spring"


def test_evaluate_prediction_matches_canonical_output():
    result = evaluate_prediction(
        prediction="Steam",
        canonical_output="steam",
        known_outputs=["mist", "steam"],
    )

    assert result.exact_canonical_match is True
    assert result.known_output_match is True
    assert result.is_empty_prediction is False


def test_evaluate_prediction_matches_known_alternative():
    result = evaluate_prediction(
        prediction="mist",
        canonical_output="steam",
        known_outputs=["mist", "steam"],
    )

    assert result.exact_canonical_match is False
    assert result.known_output_match is True


def test_evaluate_prediction_treats_empty_prediction_as_no_match():
    result = evaluate_prediction(
        prediction="   ",
        canonical_output="steam",
        known_outputs=["mist", "steam"],
    )

    assert result.exact_canonical_match is False
    assert result.known_output_match is False
    assert result.is_empty_prediction is True
