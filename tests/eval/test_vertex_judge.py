import json
from types import SimpleNamespace
from unittest.mock import patch

from src.eval.vertex_judge import (
    VertexGenAINoveltyJudge,
    normalize_vertex_model_name,
    parse_novelty_batch_judge_response,
    parse_novelty_judge_response,
    resolve_vertex_judge_backend,
)


def test_parse_novelty_judge_response_reads_json():
    response = parse_novelty_judge_response(
        json.dumps({"novelty_score": 0.73, "reason": "Distinct from the references."})
    )

    assert response.novelty_score == 0.73
    assert response.reason == "Distinct from the references."


def test_parse_novelty_judge_response_falls_back_to_float():
    response = parse_novelty_judge_response("novelty_score: 0.25")

    assert response.novelty_score == 0.25


def test_parse_novelty_batch_judge_response_reads_json_list():
    response = parse_novelty_batch_judge_response(
        json.dumps(
            {
                "results": [
                    {"output": "steam engine", "novelty_score": 0.82, "reason": "New but coherent."},
                    {"output": "mist machine", "novelty_score": 0.41, "reason": "Closer to references."},
                ]
            }
        ),
        ["steam engine", "mist machine"],
    )

    assert [result.novelty_score for result in response.results] == [0.82, 0.41]
    assert [result.reason for result in response.results] == ["New but coherent.", "Closer to references."]


def test_parse_novelty_batch_judge_response_tolerates_rewritten_output_field():
    response = parse_novelty_batch_judge_response(
        json.dumps(
            {
                "results": [
                    {"output": "steam engine variant", "novelty_score": 0.82, "reason": "New but coherent."},
                    {"output": "mist machine alt", "novelty_score": 0.41, "reason": "Closer to references."},
                ]
            }
        ),
        ["steam engine", "mist machine"],
    )

    assert [result.novelty_score for result in response.results] == [0.82, 0.41]


def test_parse_novelty_batch_judge_response_truncates_extra_items():
    response = parse_novelty_batch_judge_response(
        json.dumps(
            {
                "results": [
                    {"output": "steam engine", "novelty_score": 0.82, "reason": "New but coherent."},
                    {"output": "mist machine", "novelty_score": 0.41, "reason": "Closer to references."},
                    {"output": "extra item", "novelty_score": 0.12, "reason": "Should be ignored."},
                ]
            }
        ),
        ["steam engine", "mist machine"],
    )

    assert [result.novelty_score for result in response.results] == [0.82, 0.41]
    assert len(response.results) == 2


def test_resolve_vertex_judge_backend_uses_anthropic_for_claude_models():
    assert resolve_vertex_judge_backend("claude-haiku-4-5@20251001") == "anthropic"


def test_resolve_vertex_judge_backend_uses_genai_for_non_claude_models():
    assert resolve_vertex_judge_backend("mistralai-mistral-small-2503") == "genai"


def test_normalize_vertex_model_name_rewrites_legacy_mistral_alias():
    assert normalize_vertex_model_name("mistralai-mistral-small-2503") == "mistralai/mistralai-mistral-small-2503"


def test_normalize_vertex_model_name_rewrites_bare_mistral_model():
    assert normalize_vertex_model_name("mistral-small-2503") == "mistralai/mistralai-mistral-small-2503"


def test_normalize_vertex_model_name_rewrites_grok_model():
    assert normalize_vertex_model_name("grok-4.1-fast-non-reasoning") == "xai/grok-4.1-fast-non-reasoning"


def test_normalize_vertex_model_name_preserves_explicit_publisher_path():
    assert (
        normalize_vertex_model_name("mistralai/mistralai-mistral-small-2503")
        == "mistralai/mistralai-mistral-small-2503"
    )


def test_vertex_genai_judge_retries_after_invalid_json():
    responses = iter(
        [
            SimpleNamespace(text='{"results":[{"output":"steam","novelty_score":0.7 "reason":"bad json"}]}'),
            SimpleNamespace(
                text=json.dumps(
                    {
                        "results": [
                            {"output": "steam", "novelty_score": 0.7, "reason": "Distinct enough."},
                        ]
                    }
                )
            ),
        ]
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: next(responses))
    )

    with patch("google.genai.Client", return_value=fake_client):
        judge = VertexGenAINoveltyJudge(
            project_id="proj",
            region="us-central1",
            model="gemini-2.5-flash-lite",
        )
        result = judge.score_batch(
            input_a="fire",
            input_b="water",
            generated_outputs=["steam"],
            recipe_candidates=["steam"],
            train_outputs=["steam"],
        )

    assert [item.novelty_score for item in result.results] == [0.7]


def test_vertex_genai_judge_retries_after_incomplete_batch():
    responses = iter(
        [
            SimpleNamespace(
                text=json.dumps(
                    {
                        "results": [
                            {"output": "steam", "novelty_score": 0.7, "reason": "Distinct enough."},
                        ]
                    }
                )
            ),
            SimpleNamespace(
                text=json.dumps(
                    {
                        "results": [
                            {"output": "steam", "novelty_score": 0.7, "reason": "Distinct enough."},
                            {"output": "mist", "novelty_score": 0.5, "reason": "Moderately novel."},
                        ]
                    }
                )
            ),
        ]
    )
    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: next(responses))
    )

    with patch("google.genai.Client", return_value=fake_client):
        judge = VertexGenAINoveltyJudge(
            project_id="proj",
            region="us-central1",
            model="gemini-2.5-flash-lite",
        )
        result = judge.score_batch(
            input_a="fire",
            input_b="water",
            generated_outputs=["steam", "mist"],
            recipe_candidates=["steam", "mist"],
            train_outputs=["steam", "mist"],
        )

    assert [item.novelty_score for item in result.results] == [0.7, 0.5]
