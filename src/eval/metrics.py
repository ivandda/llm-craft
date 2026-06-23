import re
from dataclasses import dataclass


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PredictionEvaluation:
    exact_canonical_match: bool
    known_output_match: bool
    is_empty_prediction: bool


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", value.strip().lower())


def evaluate_prediction(
    prediction: str | None,
    canonical_output: str,
    known_outputs: list[str],
) -> PredictionEvaluation:
    normalized_prediction = normalize_answer(prediction)
    normalized_canonical = normalize_answer(canonical_output)
    normalized_known_outputs = {normalize_answer(output) for output in known_outputs}

    is_empty = normalized_prediction == ""
    return PredictionEvaluation(
        exact_canonical_match=not is_empty and normalized_prediction == normalized_canonical,
        known_output_match=not is_empty and normalized_prediction in normalized_known_outputs,
        is_empty_prediction=is_empty,
    )
