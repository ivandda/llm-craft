import json

from src.eval.vertex_judge import parse_novelty_batch_judge_response, parse_novelty_judge_response


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
