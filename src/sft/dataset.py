from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Candidate:
    output: str
    source: str | None = None
    rank: int | None = None
    rationale: str | None = None
    weight: float | None = None


@dataclass(frozen=True)
class RecipeExample:
    input_a: str
    input_b: str
    candidates: list[Candidate]


def candidate_from_mapping(data: dict[str, Any], weight_field: str) -> Candidate:
    output = data.get("output")
    if not output:
        raise ValueError("Each candidate must contain a non-empty `output` field.")
    raw_rank = data.get("rank")
    rank = int(raw_rank) if raw_rank is not None else None
    raw_weight = data.get(weight_field)
    weight = float(raw_weight) if raw_weight is not None else None
    return Candidate(
        output=str(output),
        source=data.get("source"),
        rank=rank,
        rationale=data.get("rationale"),
        weight=weight,
    )


def normalize_candidate_weights(
    candidates: list[Candidate],
    weight_field: str = "weight",
    fallback: str = "inverse_rank",
) -> list[float]:
    if not candidates:
        raise ValueError("Cannot normalize weights for an empty candidate list.")

    explicit_weights = [candidate.weight for candidate in candidates]
    if all(weight is not None for weight in explicit_weights):
        weights = [float(weight) for weight in explicit_weights if weight is not None]
    elif fallback == "uniform":
        weights = [1.0 for _ in candidates]
    elif fallback == "inverse_rank":
        weights = [1.0 / float(candidate.rank or index + 1) for index, candidate in enumerate(candidates)]
    else:
        raise ValueError(f"Unsupported weight fallback: {fallback}")

    if any(weight < 0 for weight in weights):
        raise ValueError(f"`{weight_field}` values must be non-negative.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Candidate weights must sum to a positive value before normalization.")
    return [weight / total for weight in weights]


class RecipeSFTDataset:
    def __init__(
        self,
        path: str | Path,
        *,
        weight_field: str = "weight",
        weight_fallback: str = "inverse_rank",
        max_examples: int | None = None,
        merge_duplicate_recipes: bool = True,
    ) -> None:
        self.path = Path(path)
        self.weight_field = weight_field
        self.weight_fallback = weight_fallback
        self.merge_duplicate_recipes = merge_duplicate_recipes
        self.examples = load_recipe_jsonl(
            self.path,
            weight_field=weight_field,
            weight_fallback=weight_fallback,
            max_examples=max_examples,
            merge_duplicate_recipes=merge_duplicate_recipes,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RecipeExample:
        return self.examples[index]


def _candidate_rows(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if isinstance(record.get("candidate_outputs"), list):
        return record["candidate_outputs"]
    if isinstance(record.get("outputs"), list):
        return [{"output": output, "rank": index + 1} for index, output in enumerate(record["outputs"]) if output]
    if record.get("output"):
        return [{"output": record["output"], "rank": 1}]
    raise ValueError("Record must contain `candidate_outputs`, `outputs`, or `output`.")


def _raw_recipe_from_record(record: dict[str, Any], *, weight_field: str) -> RecipeExample:
    input_a = record.get("input_a")
    input_b = record.get("input_b")
    if not input_a or not input_b:
        raise ValueError("Each JSONL record must contain non-empty `input_a` and `input_b` fields.")

    candidates = [candidate_from_mapping(row, weight_field) for row in _candidate_rows(record)]
    if not candidates:
        raise ValueError("Each recipe must contain at least one candidate output.")
    return RecipeExample(input_a=str(input_a), input_b=str(input_b), candidates=candidates)


def _with_normalized_weights(
    recipe: RecipeExample,
    *,
    weight_field: str,
    weight_fallback: str,
) -> RecipeExample:
    normalized = normalize_candidate_weights(
        recipe.candidates,
        weight_field=weight_field,
        fallback=weight_fallback,
    )
    candidates = [
        Candidate(
            output=candidate.output,
            source=candidate.source,
            rank=candidate.rank,
            rationale=candidate.rationale,
            weight=weight,
        )
        for candidate, weight in zip(recipe.candidates, normalized, strict=True)
    ]
    return RecipeExample(input_a=recipe.input_a, input_b=recipe.input_b, candidates=candidates)


def merge_recipe_examples(
    recipes: list[RecipeExample],
    *,
    weight_field: str,
    weight_fallback: str,
) -> list[RecipeExample]:
    merged: OrderedDict[tuple[str, str], list[Candidate]] = OrderedDict()
    for recipe in recipes:
        merged.setdefault((recipe.input_a, recipe.input_b), []).extend(recipe.candidates)

    normalized_recipes: list[RecipeExample] = []
    for (input_a, input_b), candidates in merged.items():
        # Deduplicate repeated outputs deterministically by preserving the first
        # candidate metadata and summing the pre-normalized mass contributed by
        # every duplicate before the final per-recipe renormalization.
        base_weights = normalize_candidate_weights(
            candidates,
            weight_field=weight_field,
            fallback=weight_fallback,
        )
        deduped: OrderedDict[str, Candidate] = OrderedDict()
        deduped_weights: OrderedDict[str, float] = OrderedDict()
        for candidate, base_weight in zip(candidates, base_weights, strict=True):
            if candidate.output not in deduped:
                deduped[candidate.output] = candidate
                deduped_weights[candidate.output] = 0.0
            deduped_weights[candidate.output] += base_weight

        merged_candidates = [
            Candidate(
                output=candidate.output,
                source=candidate.source,
                rank=candidate.rank,
                rationale=candidate.rationale,
                weight=deduped_weights[output],
            )
            for output, candidate in deduped.items()
        ]
        normalized_recipes.append(
            _with_normalized_weights(
                RecipeExample(input_a=input_a, input_b=input_b, candidates=merged_candidates),
                weight_field=weight_field,
                weight_fallback=weight_fallback,
            )
        )
    return normalized_recipes


def load_recipe_jsonl(
    path: str | Path,
    *,
    weight_field: str = "weight",
    weight_fallback: str = "inverse_rank",
    max_examples: int | None = None,
    merge_duplicate_recipes: bool = True,
) -> list[RecipeExample]:
    examples: list[RecipeExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not merge_duplicate_recipes and max_examples is not None and len(examples) >= max_examples:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                examples.append(_raw_recipe_from_record(record, weight_field=weight_field))
            except Exception as exc:
                raise ValueError(f"Invalid recipe record at {path}:{line_number}: {exc}") from exc
    if merge_duplicate_recipes:
        examples = merge_recipe_examples(
            examples,
            weight_field=weight_field,
            weight_fallback=weight_fallback,
        )
        if max_examples is not None:
            examples = examples[:max_examples]
        return examples
    return [
        _with_normalized_weights(
            recipe,
            weight_field=weight_field,
            weight_fallback=weight_fallback,
        )
        for recipe in examples
    ]
