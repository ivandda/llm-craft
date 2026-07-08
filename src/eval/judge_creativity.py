"""LLM-judge creativity scoring for generated recipe predictions.

Coverage (score_coverage.py) tells us whether a prediction matches the teacher's
known_outputs. It is blind to correct-but-off-label answers: when the student
produces a plausible result the teacher never listed (e.g. element+skull ->
"calcium phosphate", better than the label "bone"), coverage scores it 0.

This module fills that gap with a strong LLM judge (Claude on Vertex, stronger
than the Gemini teacher). It scores two INDEPENDENT axes and reports each as its
own rate, never collapsing them into a single opaque number:

    plausible?  -> the judge decides, REFERENCE-FREE: "is <output> a sensible
                   result of combining A + B?" It never sees the teacher's
                   known_outputs, so it cannot just rubber-stamp matches nor
                   penalise a valid off-label answer.
    in-dataset? -> deterministic code: does the output match a known_output
                   (exact-normalized or semantic cosine >= threshold)?

Buckets (per sampled output):
    invalid      = not plausible                    -> wrong / malformed
    valid-known  = plausible AND in-dataset         -> correct, but not new
    valid-novel  = plausible AND off-dataset        -> DISCOVERY (correct + new)

Headline rates (over N recipes), for both the top-1 sample and any@k:
    Validity   = (valid-known + valid-novel) / N    -> how often correct
    Creativity =  valid-novel / N                   -> how often correct AND new

Generation is the expensive GPU step and is already saved to predictions.jsonl.
This scorer runs offline against the judge API, so it re-scores any saved run
without re-running the student.

Usage (realtime):
    uv run --group vertex python -m src.eval.judge_creativity \
        gs://llm-craft-bucket/eval_outputs/cu126_softce_test/predictions.jsonl \
        --model claude-sonnet-5 --output_dir runs/judge/softce \
        --semantic_threshold 0.75 [--max_examples 20] [--concurrency 8]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.eval.metrics import normalize_answer

PLAUSIBILITY_SYSTEM_PROMPT = (
    "You are an expert judge for a compositional creativity game inspired by Infinite "
    "Craft: two input concepts are combined into a resulting concept (fire + water -> "
    "steam). Judge ONLY plausibility: could a reasonable person accept the generated "
    "output as a sensible result of combining the two inputs? "
    "Say plausible=true when the output is a coherent, on-topic concept that follows "
    "from the two inputs (a real or imaginable thing, a valid association, cause, "
    "product, or blend). Say plausible=false when it is wrong, off-topic, self-"
    "contradictory, empty, or malformed (garbled tokens, cut-off words, stray "
    "punctuation, or meta-text like 'Hmm the user'). "
    "Do NOT reward or penalise popularity, novelty, length, or whether you have seen "
    "the answer before; a valid answer you did not expect is still plausible. "
    "Return STRICT JSON: a top-level key 'results' with one item per generated output, "
    "each item having keys 'output' (copied verbatim), 'plausible' (true/false), "
    "'plausibility' (float 0..1 confidence that it is plausible), and 'reason' (short "
    "string). Return items in the same order as the inputs."
)


@dataclass(frozen=True)
class Verdict:
    """One judge decision for one generated output."""

    plausible: bool
    plausibility: float
    reason: str


class PlausibilityJudge(Protocol):
    """Anything that can score a recipe's outputs. Injectable for offline tests."""

    def score_recipe(self, input_a: str, input_b: str, outputs: Sequence[str]) -> list[Verdict]:
        ...


# --------------------------------------------------------------------------- #
# Membership (deterministic): is an output already in the teacher's answers?
# --------------------------------------------------------------------------- #
def _exact_in_known(output: str, known_normalized: set[str]) -> bool:
    return normalize_answer(output) in known_normalized


def _semantic_in_known(output: str, known_outputs: list[str], embedder: Any, threshold: float) -> bool:
    import numpy as np

    if not known_outputs or not normalize_answer(output):
        return False
    out_vec = np.asarray(embedder.encode([output]), dtype=float)[0]
    known_matrix = np.asarray(embedder.encode(known_outputs), dtype=float)
    out_norm = np.linalg.norm(out_vec)
    if out_norm == 0 or known_matrix.size == 0:
        return False
    known_norms = np.linalg.norm(known_matrix, axis=1)
    known_norms[known_norms == 0] = 1.0
    cosines = (known_matrix @ out_vec) / (known_norms * out_norm)
    return bool(cosines.max() >= threshold)


def is_in_dataset(
    output: str,
    known_outputs: list[str],
    known_normalized: set[str],
    embedder: Any | None,
    threshold: float,
) -> bool:
    """Match against known_outputs by exact-normalized string, then (optionally) semantic cosine."""
    if _exact_in_known(output, known_normalized):
        return True
    if embedder is not None:
        return _semantic_in_known(output, known_outputs, embedder, threshold)
    return False


# --------------------------------------------------------------------------- #
# Bucketing + aggregation
# --------------------------------------------------------------------------- #
BUCKETS = ("invalid", "valid_known", "valid_novel")


def bucket_of(plausible: bool, in_dataset: bool) -> str:
    if not plausible:
        return "invalid"
    return "valid_known" if in_dataset else "valid_novel"


def judge_record(
    record: dict[str, Any],
    judge: PlausibilityJudge,
    embedder: Any | None,
    threshold: float,
) -> dict[str, Any]:
    """Judge one prediction record and return per-sample buckets + recipe-level flags."""
    samples: list[str] = record.get("sampled_outputs") or [record.get("prediction", "")]
    known_outputs: list[str] = record.get("known_outputs") or []
    known_normalized = {normalize_answer(o) for o in known_outputs}

    # Judge distinct non-empty outputs once (mode collapse repeats the same string).
    distinct = list(dict.fromkeys(s for s in samples if normalize_answer(s)))
    verdict_by_output: dict[str, Verdict] = {}
    if distinct:
        verdicts = judge.score_recipe(record.get("input_a", ""), record.get("input_b", ""), distinct)
        for out, verdict in zip(distinct, verdicts):
            verdict_by_output[out] = verdict

    per_sample: list[dict[str, Any]] = []
    for sample in samples:
        verdict = verdict_by_output.get(sample)
        if verdict is None:  # empty string or judged as not present -> treat as invalid
            verdict = Verdict(plausible=False, plausibility=0.0, reason="empty or unjudged output")
        in_ds = is_in_dataset(sample, known_outputs, known_normalized, embedder, threshold)
        per_sample.append(
            {
                "output": sample,
                "plausible": verdict.plausible,
                "plausibility": verdict.plausibility,
                "in_dataset": in_ds,
                "bucket": bucket_of(verdict.plausible, in_ds),
                "reason": verdict.reason,
            }
        )

    top1 = per_sample[0]
    valid_top1 = bool(top1["plausible"])
    creative_top1 = bool(top1["plausible"] and not top1["in_dataset"])
    valid_anyk = any(s["plausible"] for s in per_sample)
    creative_anyk = any(s["plausible"] and not s["in_dataset"] for s in per_sample)

    return {
        "pair_id": record.get("pair_id"),
        "input_a": record.get("input_a"),
        "input_b": record.get("input_b"),
        "known_outputs": known_outputs,
        "samples": per_sample,
        "valid_top1": valid_top1,
        "creative_top1": creative_top1,
        "valid_anyk": valid_anyk,
        "creative_anyk": creative_anyk,
    }


def aggregate(judged: list[dict[str, Any]]) -> dict[str, float]:
    n = len(judged)
    if n == 0:
        return {"n": 0}
    top1_buckets = {b: 0 for b in BUCKETS}
    for j in judged:
        top1_buckets[j["samples"][0]["bucket"]] += 1
    return {
        "n": n,
        "validity_top1": sum(j["valid_top1"] for j in judged) / n,
        "creativity_top1": sum(j["creative_top1"] for j in judged) / n,
        "validity_anyk": sum(j["valid_anyk"] for j in judged) / n,
        "creativity_anyk": sum(j["creative_anyk"] for j in judged) / n,
        "top1_invalid_rate": top1_buckets["invalid"] / n,
        "top1_valid_known_rate": top1_buckets["valid_known"] / n,
        "top1_valid_novel_rate": top1_buckets["valid_novel"] / n,
    }


def score_records(
    records: list[dict[str, Any]],
    judge: PlausibilityJudge,
    embedder: Any | None,
    threshold: float,
    concurrency: int = 1,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    def _one(record: dict[str, Any]) -> dict[str, Any]:
        return judge_record(record, judge, embedder, threshold)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            judged = list(pool.map(_one, records))
    else:
        judged = [_one(r) for r in records]
    return aggregate(judged), judged


# --------------------------------------------------------------------------- #
# Real judge: Claude on Vertex (reference-free plausibility)
# --------------------------------------------------------------------------- #
class VertexAnthropicPlausibilityJudge:
    def __init__(self, *, project_id: str, region: str, model: str) -> None:
        from anthropic import AnthropicVertex

        from src.eval.vertex_judge import normalize_vertex_model_name

        self.client = AnthropicVertex(project_id=project_id, region=region)
        self.model = normalize_vertex_model_name(model)

    def score_recipe(self, input_a: str, input_b: str, outputs: Sequence[str]) -> list[Verdict]:
        payload = {
            "input_a": input_a,
            "input_b": input_b,
            "generated_outputs": list(outputs),
            "instructions": {"task": "Judge plausibility only", "scale": "plausible bool + 0..1 confidence"},
        }
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max(256, 96 * len(outputs)),
            temperature=0,
            system=PLAUSIBILITY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        text = "\n".join(
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", "") == "text" and getattr(block, "text", "")
        )
        return parse_plausibility_response(text, list(outputs))


class VertexGenAIPlausibilityJudge:
    """Gemini judge via google-genai. Works today on this project (Claude is not
    enabled in its Vertex Model Garden). Use a model stronger than the teacher
    (gemini-2.5-flash), e.g. gemini-2.5-pro."""

    def __init__(self, *, project_id: str, region: str, model: str) -> None:
        from google import genai

        from src.eval.vertex_judge import normalize_vertex_model_name

        self.client = genai.Client(vertexai=True, project=project_id, location=region)
        self.model = normalize_vertex_model_name(model)
        self.max_parse_retries = 2

    def score_recipe(self, input_a: str, input_b: str, outputs: Sequence[str]) -> list[Verdict]:
        from google.genai import types

        payload = {
            "input_a": input_a,
            "input_b": input_b,
            "generated_outputs": list(outputs),
            "instructions": {"task": "Judge plausibility only", "scale": "plausible bool + 0..1 confidence"},
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_parse_retries + 2):
            response = self.client.models.generate_content(
                model=self.model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=PLAUSIBILITY_SYSTEM_PROMPT,
                    temperature=0,
                    # 2.5 models spend tokens on hidden thinking; keep the ceiling high
                    # enough that the JSON answer is not truncated to empty.
                    max_output_tokens=max(2048, 256 * len(outputs)),
                    response_mime_type="application/json",
                ),
            )
            text = getattr(response, "text", "") or ""
            if text:
                try:
                    return parse_plausibility_response(text, list(outputs))
                except Exception as exc:  # malformed/short JSON -> retry
                    last_error = exc
            else:
                last_error = ValueError("Vertex GenAI judge returned no text.")
        assert last_error is not None
        raise last_error


DEFAULT_GEMINI_REGION = "us-central1"
DEFAULT_ANTHROPIC_REGION = "us-east5"  # Claude on Vertex is region-restricted.


def resolve_judge_region(model: str, region: str) -> str:
    """Anthropic Claude models are not served from us-central1 (the Gemini default).
    Route them to us-east5 unless the caller picked a non-default region on purpose."""
    from src.eval.vertex_judge import resolve_vertex_judge_backend

    if resolve_vertex_judge_backend(model) == "anthropic" and region == DEFAULT_GEMINI_REGION:
        return DEFAULT_ANTHROPIC_REGION
    return region


def build_plausibility_judge(*, project_id: str, region: str, model: str):
    from src.eval.vertex_judge import resolve_vertex_judge_backend

    region = resolve_judge_region(model, region)
    if resolve_vertex_judge_backend(model) == "anthropic":
        return VertexAnthropicPlausibilityJudge(project_id=project_id, region=region, model=model)
    return VertexGenAIPlausibilityJudge(project_id=project_id, region=region, model=model)


def parse_plausibility_response(text: str, expected_outputs: list[str]) -> list[Verdict]:
    """Parse the judge's strict-JSON 'results' list into one Verdict per output, by position."""
    from src.eval.vertex_judge import extract_json_object

    payload = extract_json_object(text)
    raw = payload.get("results")
    if not isinstance(raw, list):
        raise ValueError("Judge response must contain a top-level 'results' list.")
    raw = raw[: len(expected_outputs)]
    if len(raw) < len(expected_outputs):
        raise ValueError(
            f"Judge returned {len(raw)} results, expected {len(expected_outputs)}."
        )
    verdicts: list[Verdict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each judge result must be a JSON object.")
        plausible = bool(item.get("plausible"))
        try:
            plausibility = float(item.get("plausibility", 1.0 if plausible else 0.0))
        except (TypeError, ValueError):
            plausibility = 1.0 if plausible else 0.0
        plausibility = min(1.0, max(0.0, plausibility))
        verdicts.append(Verdict(plausible=plausible, plausibility=plausibility, reason=str(item.get("reason", ""))))
    return verdicts


# --------------------------------------------------------------------------- #
# Gemini batch mode (Vertex BatchPredictionJob via google-genai) — ~50% cheaper.
# Claude batch is NOT available on Vertex, so batch is Gemini-only. Mirrors the
# proven teacher-enrichment batch flow (src/data/enrich_teacher.py).
# --------------------------------------------------------------------------- #
COMPLETED_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PAUSED",
}
BATCH_ID_MARKER = "JUDGE_RECIPE_KEY:"


def _normalize_batch_state(job: Any) -> str:
    """job.state may be a JobState enum (repr 'JobState.JOB_STATE_RUNNING') or a plain
    string; return the bare 'JOB_STATE_*' token either way."""
    raw = getattr(job, "state", "")
    name = getattr(raw, "name", None)
    if name:
        return str(name)
    return str(raw).rsplit(".", 1)[-1]


def _recipe_key(input_a: str, input_b: str) -> str:
    return f"{normalize_answer(input_a)}|{normalize_answer(input_b)}"


def build_batch_request_row(input_a: str, input_b: str, distinct_outputs: list[str]) -> dict[str, Any]:
    """One Vertex-batch request row for a recipe. The system prompt is folded into
    the text (Vertex batch rows carry a single content payload) and the recipe key
    is embedded so the (order-independent) output can be matched back."""
    payload = {
        "input_a": input_a,
        "input_b": input_b,
        "generated_outputs": distinct_outputs,
        "instructions": {"task": "Judge plausibility only", "scale": "plausible bool + 0..1 confidence"},
    }
    prompt = (
        f"{PLAUSIBILITY_SYSTEM_PROMPT}\n\n{BATCH_ID_MARKER} {_recipe_key(input_a, input_b)}\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "maxOutputTokens": max(2048, 256 * max(1, len(distinct_outputs))),
            },
        }
    }


def build_batch_requests(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Return (request_rows, index) where index maps recipe_key -> distinct outputs (in order)."""
    rows: list[dict[str, Any]] = []
    index: dict[str, list[str]] = {}
    for record in records:
        samples: list[str] = record.get("sampled_outputs") or [record.get("prediction", "")]
        distinct = list(dict.fromkeys(s for s in samples if normalize_answer(s)))
        if not distinct:
            continue
        key = _recipe_key(record.get("input_a", ""), record.get("input_b", ""))
        index[key] = distinct
        rows.append(build_batch_request_row(record.get("input_a", ""), record.get("input_b", ""), distinct))
    return rows, index


def _iter_batch_text_parts(payload: dict[str, Any]):
    contents = payload.get("contents") or []
    candidates = payload.get("candidates") or []
    if candidates:
        contents = [c.get("content", {}) for c in candidates]
    for content in contents:
        for part in content.get("parts", []):
            if part.get("text"):
                yield part["text"]


def parse_batch_outputs(output_rows: list[dict[str, Any]], index: dict[str, list[str]]) -> dict[str, dict[str, Verdict]]:
    """Map recipe_key -> {output_string -> Verdict} from Vertex batch output rows."""
    verdicts_by_key: dict[str, dict[str, Verdict]] = {}
    for row in output_rows:
        key = None
        for text in _iter_batch_text_parts(row.get("request", {})):
            match = re.search(rf"{re.escape(BATCH_ID_MARKER)}\s*(\S.*)", text)
            if match:
                key = match.group(1).strip()
                break
        if key is None or key not in index:
            continue
        distinct = index[key]
        if row.get("status"):  # row-level error -> leave unjudged (defaults to invalid downstream)
            continue
        response_text = next(_iter_batch_text_parts(row.get("response", {})), "")
        if not response_text:
            continue
        try:
            verdicts = parse_plausibility_response(response_text, distinct)
        except Exception:
            continue
        verdicts_by_key[key] = {out: v for out, v in zip(distinct, verdicts)}
    return verdicts_by_key


class PrecomputedJudge:
    """A judge whose verdicts were computed offline (e.g. by a batch job)."""

    def __init__(self, verdicts_by_key: dict[str, dict[str, Verdict]]) -> None:
        self.verdicts_by_key = verdicts_by_key

    def score_recipe(self, input_a: str, input_b: str, outputs: Sequence[str]) -> list[Verdict]:
        table = self.verdicts_by_key.get(_recipe_key(input_a, input_b), {})
        return [
            table.get(o, Verdict(plausible=False, plausibility=0.0, reason="missing batch verdict"))
            for o in outputs
        ]


def _gsutil_write(uri: str, text: str) -> None:
    proc = subprocess.run(["gsutil", "cp", "-", uri], input=text, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gsutil cp to {uri} failed.")


def _gsutil_ls_jsonl(prefix: str) -> list[str]:
    completed = subprocess.run(["gsutil", "ls", "-r", prefix], capture_output=True, text=True)
    return [ln.strip() for ln in completed.stdout.splitlines() if ln.strip().endswith(".jsonl")]


def _gsutil_cat_jsonl(uri: str) -> list[dict[str, Any]]:
    completed = subprocess.run(["gsutil", "cat", uri], capture_output=True, text=True, check=True)
    return [json.loads(ln) for ln in completed.stdout.splitlines() if ln.strip()]


def run_gemini_batch(
    records: list[dict[str, Any]],
    *,
    model: str,
    project_id: str,
    location: str,
    gcs_staging: str,
    label: str,
    poll_seconds: int,
) -> dict[str, dict[str, Verdict]]:
    """Export -> upload -> submit -> poll -> download -> parse. Returns verdict map."""
    from google import genai
    from google.genai import types

    rows, index = build_batch_requests(records)
    staging = gcs_staging.rstrip("/") + f"/{label}"
    input_uri = f"{staging}/input/requests.jsonl"
    output_prefix = f"{staging}/output"
    print(f"[judge-batch] uploading {len(rows)} requests -> {input_uri}", flush=True)
    _gsutil_write(input_uri, "\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    client = genai.Client(vertexai=True, project=project_id, location=location)
    job = client.batches.create(
        model=model,
        src=input_uri,
        config=types.CreateBatchJobConfig(display_name=f"judge-{label}", dest=output_prefix),
    )
    name = job.name
    print(f"[judge-batch] submitted {name}; polling every {poll_seconds}s ...", flush=True)
    state = _normalize_batch_state(job)
    while state not in COMPLETED_BATCH_STATES:
        time.sleep(poll_seconds)
        job = client.batches.get(name=name)
        state = _normalize_batch_state(job)
        print(f"[judge-batch]   state={state}", flush=True)
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job {name} ended in {state}.")

    output_files = _gsutil_ls_jsonl(output_prefix)
    print(f"[judge-batch] reading {len(output_files)} output file(s) from {output_prefix}", flush=True)
    output_rows: list[dict[str, Any]] = []
    for uri in output_files:
        output_rows.extend(_gsutil_cat_jsonl(uri))
    return parse_batch_outputs(output_rows, index)


# --------------------------------------------------------------------------- #
# I/O + CLI
# --------------------------------------------------------------------------- #
def load_prediction_records(path: str) -> list[dict[str, Any]]:
    if path.startswith("gs://"):
        completed = subprocess.run(["gsutil", "cat", path], capture_output=True, text=True, check=True)
        lines = completed.stdout.splitlines()
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _write_output(path: str, text: str) -> None:
    if path.startswith("gs://"):
        proc = subprocess.run(["gsutil", "cp", "-", path], input=text, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"gsutil cp to {path} failed.")
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")


def format_summary(label: str, scores: dict[str, float]) -> str:
    if scores.get("n", 0) == 0:
        return f"{label}: no records."
    return (
        f"{label}  (n={int(scores['n'])})\n"
        f"  Validity   top1={scores['validity_top1']:.3f}  any@k={scores['validity_anyk']:.3f}\n"
        f"  Creativity top1={scores['creativity_top1']:.3f}  any@k={scores['creativity_anyk']:.3f}\n"
        f"  top1 buckets: invalid={scores['top1_invalid_rate']:.3f}  "
        f"valid-known={scores['top1_valid_known_rate']:.3f}  "
        f"valid-novel={scores['top1_valid_novel_rate']:.3f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="predictions.jsonl path (local or gs://).")
    parser.add_argument("--label", default=None, help="Label for the printed summary.")
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Vertex judge model (must beat the gemini-2.5-flash teacher). gemini-2.5-pro is "
        "the default (supports batch). Claude ids (claude-sonnet-4-5@...) auto-route to us-east5 "
        "but need Anthropic enabled in Model Garden and only run in --mode realtime.",
    )
    parser.add_argument("--project_id", default="nlp2026-498021")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument(
        "--mode",
        choices=("realtime", "batch"),
        default="realtime",
        help="realtime = one API call per recipe (concurrent). batch = one Vertex "
        "BatchPredictionJob (~50%% cheaper, Gemini only; Claude batch is unsupported on Vertex).",
    )
    parser.add_argument(
        "--gcs_staging",
        default="gs://llm-craft-bucket/judge_batch",
        help="GCS prefix for batch input/output (mode=batch only).",
    )
    parser.add_argument("--poll_seconds", type=int, default=30, help="Batch job poll interval (mode=batch only).")
    parser.add_argument("--output_dir", default=None, help="Where to write summary.json + judgments.jsonl (local or gs://).")
    parser.add_argument("--max_examples", type=int, default=None, help="Judge only the first N recipes (pilot).")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel judge calls (mode=realtime only).")
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=0.75,
        help="Membership: an output counts as in-dataset if cosine to any known_output >= this. "
        "Set <=0 to use exact-string membership only (no embedder).",
    )
    parser.add_argument("--embedding_backend", default="sentence_embeddings")
    parser.add_argument("--embedding_model_path", default=None)
    parser.add_argument("--sentence_embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding_device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    from src.eval.vertex_judge import load_vertex_environment

    load_vertex_environment(repo_root)

    records = load_prediction_records(args.predictions)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    embedder = None
    if args.semantic_threshold and args.semantic_threshold > 0:
        from src.eval.embeddings import build_text_embedder

        embedder = build_text_embedder(
            args.embedding_backend,
            word_vector_path=args.embedding_model_path,
            sentence_transformer_model=args.sentence_embedding_model,
            device=args.embedding_device,
        )

    label = args.label or Path(args.predictions).parent.name or "predictions"

    if args.mode == "batch":
        from src.eval.vertex_judge import resolve_vertex_judge_backend

        if resolve_vertex_judge_backend(args.model) == "anthropic":
            raise SystemExit(
                "Batch mode is Gemini-only: the Anthropic Batch API is not supported on the "
                "Vertex client. Use --mode realtime for Claude, or --model gemini-2.5-pro for batch."
            )
        print(f"[judge] batch-scoring {len(records)} recipes with {args.model} ...", flush=True)
        verdicts_by_key = run_gemini_batch(
            records,
            model=args.model,
            project_id=args.project_id,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            gcs_staging=args.gcs_staging,
            label=label,
            poll_seconds=args.poll_seconds,
        )
        judge: PlausibilityJudge = PrecomputedJudge(verdicts_by_key)
        scores, judged = score_records(records, judge, embedder, args.semantic_threshold, concurrency=1)
    else:
        judge = build_plausibility_judge(
            project_id=args.project_id, region=args.region, model=args.model
        )
        print(f"[judge] scoring {len(records)} recipes with {args.model} ...", flush=True)
        scores, judged = score_records(records, judge, embedder, args.semantic_threshold, args.concurrency)
    print(format_summary(label, scores))

    if args.output_dir:
        summary = {"label": label, "model": args.model, "predictions": args.predictions, **scores}
        _write_output(args.output_dir.rstrip("/") + "/summary.json", json.dumps(summary, indent=2))
        _write_output(
            args.output_dir.rstrip("/") + "/judgments.jsonl",
            "\n".join(json.dumps(j, ensure_ascii=False) for j in judged),
        )
        print(f"[judge] wrote summary.json + judgments.jsonl to {args.output_dir}")


if __name__ == "__main__":
    main()
