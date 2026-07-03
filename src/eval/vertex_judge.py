from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


NOVELTY_SYSTEM_PROMPT = (
    "You are an expert judge for a compositional creativity benchmark inspired by Infinite Craft. "
    "Score only novelty. Do not score plausibility, grammar, or popularity. "
    "A novelty score of 0.0 means the generated output is an exact duplicate, trivial variant, or near-paraphrase "
    "of the provided references. A novelty score of 1.0 means the generated output is clearly distinct from the "
    "references while still being a meaningful composition of the two inputs. "
    "Return strict JSON with a top-level key results containing one item per generated output. "
    "Each item must have keys output, novelty_score, and reason. novelty_score must be between 0 and 1."
)


@dataclass(frozen=True)
class NoveltyJudgeResult:
    novelty_score: float
    reason: str
    raw_text: str


@dataclass(frozen=True)
class NoveltyBatchJudgeResult:
    results: list[NoveltyJudgeResult]
    raw_text: str


def load_vertex_environment(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    creds_rel_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_rel_path:
        creds_path = Path(creds_rel_path)
        if not creds_path.is_absolute():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((repo_root / creds_path).resolve())


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Judge response does not contain JSON: {text!r}")
    return json.loads(text[start : end + 1])


def parse_novelty_judge_response(text: str) -> NoveltyJudgeResult:
    try:
        payload = extract_json_object(text)
        novelty_score = float(payload["novelty_score"])
        reason = str(payload.get("reason", ""))
    except Exception:
        match = re.search(r"([01](?:\.\d+)?)", text)
        if match is None:
            raise
        novelty_score = float(match.group(1))
        reason = text.strip()
    novelty_score = min(1.0, max(0.0, novelty_score))
    return NoveltyJudgeResult(novelty_score=novelty_score, reason=reason, raw_text=text)


def parse_novelty_batch_judge_response(text: str, expected_outputs: list[str]) -> NoveltyBatchJudgeResult:
    payload = extract_json_object(text)
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Judge response must contain a top-level 'results' list.")
    if len(raw_results) != len(expected_outputs):
        raise ValueError(
            f"Judge returned {len(raw_results)} results, but {len(expected_outputs)} outputs were expected."
        )

    parsed_results: list[NoveltyJudgeResult] = []
    for expected_output, item in zip(expected_outputs, raw_results):
        if not isinstance(item, dict):
            raise ValueError("Each novelty judge result must be a JSON object.")
        novelty_score = min(1.0, max(0.0, float(item["novelty_score"])))
        reason = str(item.get("reason", ""))
        returned_output = str(item.get("output", ""))
        if returned_output and returned_output != expected_output:
            raise ValueError(
                f"Judge output order mismatch. Expected {expected_output!r}, received {returned_output!r}."
            )
        parsed_results.append(
            NoveltyJudgeResult(
                novelty_score=novelty_score,
                reason=reason,
                raw_text=text,
            )
        )
    return NoveltyBatchJudgeResult(results=parsed_results, raw_text=text)


class VertexAnthropicNoveltyJudge:
    def __init__(
        self,
        *,
        project_id: str,
        region: str,
        model: str,
    ) -> None:
        from anthropic import AnthropicVertex

        self.client = AnthropicVertex(project_id=project_id, region=region)
        self.model = model
        self.cache: dict[
            tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
            NoveltyBatchJudgeResult,
        ] = {}

    def score_batch(
        self,
        *,
        input_a: str,
        input_b: str,
        generated_outputs: list[str],
        recipe_candidates: list[str],
        train_outputs: list[str],
    ) -> NoveltyBatchJudgeResult:
        cache_key = (
            input_a,
            input_b,
            tuple(recipe_candidates),
            tuple(generated_outputs),
            tuple(train_outputs),
        )
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt_payload = {
            "input_a": input_a,
            "input_b": input_b,
            "generated_outputs": generated_outputs,
            "recipe_candidates": recipe_candidates,
            "train_outputs_for_same_pair": train_outputs,
            "instructions": {
                "task": "Score novelty only",
                "scale": "0.0 to 1.0",
                "return_format": {
                    "results": [
                        {
                            "output": "string copied from generated_outputs",
                            "novelty_score": "float between 0 and 1",
                            "reason": "short string",
                        }
                    ],
                },
            },
        }
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max(256, 96 * len(generated_outputs)),
            temperature=0,
            system=NOVELTY_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload, ensure_ascii=False),
                }
            ],
        )
        text_blocks = [
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", "") == "text" and getattr(block, "text", "")
        ]
        if not text_blocks:
            raise ValueError("Vertex judge returned no text blocks.")
        result = parse_novelty_batch_judge_response("\n".join(text_blocks), generated_outputs)
        self.cache[cache_key] = result
        return result
