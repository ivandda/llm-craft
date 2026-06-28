import json

import pytest

from src.sft.dataset import Candidate, RecipeSFTDataset, load_recipe_jsonl, normalize_candidate_weights


def test_load_recipe_jsonl_reads_candidate_outputs(tmp_path):
    path = tmp_path / "recipes.jsonl"
    path.write_text(
        json.dumps(
            {
                "input_a": "fire",
                "input_b": "water",
                "candidate_outputs": [
                    {"output": "steam", "source": "observed", "rank": 1},
                    {"output": "vapor", "source": "teacher", "rank": 2},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_recipe_jsonl(path)

    assert len(examples) == 1
    assert examples[0].input_a == "fire"
    assert [candidate.output for candidate in examples[0].candidates] == ["steam", "vapor"]


def test_normalize_candidate_weights_uniform():
    candidates = [Candidate(output="a", rank=1), Candidate(output="b", rank=9)]

    weights = normalize_candidate_weights(candidates, fallback="uniform")

    assert weights == [0.5, 0.5]


def test_normalize_candidate_weights_inverse_rank_default():
    candidates = [Candidate(output="a", rank=1), Candidate(output="b", rank=2)]

    weights = normalize_candidate_weights(candidates, fallback="inverse_rank")

    assert weights == pytest.approx([2 / 3, 1 / 3])
    assert sum(weights) == pytest.approx(1.0)


def test_dataset_limits_examples(tmp_path):
    path = tmp_path / "recipes.jsonl"
    rows = [
        {"input_a": "a", "input_b": "b", "candidate_outputs": [{"output": "c", "rank": 1}]},
        {"input_a": "d", "input_b": "e", "candidate_outputs": [{"output": "f", "rank": 1}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = RecipeSFTDataset(path, max_examples=1)

    assert len(dataset) == 1
