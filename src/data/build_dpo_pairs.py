"""Build a DPO preference dataset from on-policy samples of the SFT model.

Rationale: DPO needs (chosen, rejected) pairs. To fix the winner's form problem
(verbosity + garble) the negatives must be the model's OWN high-probability mistakes,
and the raw text (NOT the eval's post-processed `sampled_outputs`, which are already
cleaned to <=3 words). So this tool generates its own RAW samples over a subset of
`datasets/final-10k/train.jsonl` and applies rule-based selection:

- **chosen**  = a short (<=2 words) valid answer. Prefer an on-policy sample the model
  itself produced; fall back to a short dataset candidate (rank-1/observed is canonical).
- **rejected** = one of the model's own samples that is verbose (>=3 words) or invalid
  (on-policy negatives). Never random — DPO only learns where the model is currently wrong.

Recipes with no usable short chosen, or where every sample is already short+valid (no
negative to push down), are skipped and counted.

Writes `pairs.jsonl` (+ `pairs.meta.json` with the build fingerprint). The pure selection
logic (`select_preference_pair`) is unit-tested without a model.

Usage (GPU, on Vertex):
    uv run --group vertex python -m src.data.build_dpo_pairs \
        --run_dir /gcs/llm-craft-bucket/runs/<soft_ce_run> \
        --input_file /gcs/llm-craft-bucket/datasets/final-10k/train.jsonl \
        --output_dir /gcs/llm-craft-bucket/dpo/softce_pairs \
        --num_samples 6 --max_examples 2000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.eval.metrics import evaluate_prediction, normalize_answer

SHORT_MAX_WORDS = 2
VERBOSE_MIN_WORDS = 3


def _word_count(text: str) -> int:
    return len(normalize_answer(text).split())


def _most_frequent(raws: list[str]) -> str:
    """Most frequent by normalized form; tie-break shortest then first-seen."""
    counts = Counter(normalize_answer(r) for r in raws)
    best_norm, _ = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])))
    for raw in raws:  # return the first raw whose normalized form is the winner
        if normalize_answer(raw) == best_norm:
            return raw
    return raws[0]


def select_preference_pair(
    input_a: str,
    input_b: str,
    known_outputs: list[str],
    canonical_output: str,
    raw_samples: list[str],
    *,
    candidate_rows: list[dict[str, Any]] | None = None,
    short_max_words: int = SHORT_MAX_WORDS,
    verbose_min_words: int = VERBOSE_MIN_WORDS,
) -> dict[str, Any] | None:
    """Apply the chosen/rejected rules to one recipe's raw samples. Returns a pair dict
    or None (with the skip reason available via the returned None — counted by caller)."""
    samples = [s for s in raw_samples if s is not None]

    def valid(text: str) -> bool:
        return evaluate_prediction(text, canonical_output, known_outputs).known_output_match

    labeled = [
        {"raw": s, "norm": normalize_answer(s), "wc": _word_count(s), "valid": valid(s)}
        for s in samples
    ]

    # --- chosen: on-policy short+valid, else short dataset candidate ---
    short_valid = [d["raw"] for d in labeled if d["valid"] and 0 < d["wc"] <= short_max_words]
    if short_valid:
        chosen = _most_frequent(short_valid)
        chosen_source = "on_policy"
    else:
        chosen = None
        chosen_source = "dataset_candidate"
        rows = candidate_rows or []
        short_candidates = [r for r in rows if r.get("output") and 0 < _word_count(r["output"]) <= short_max_words]
        if short_candidates:
            # prefer observed / rank-1, else the lowest rank.
            def _rank_key(row: dict[str, Any]) -> tuple[int, int]:
                observed = 0 if row.get("source") == "observed" else 1
                return (observed, int(row.get("rank", 999)))

            chosen = min(short_candidates, key=_rank_key)["output"]
    if not chosen or not normalize_answer(chosen):
        return None  # skip: no usable short chosen

    chosen_norm = normalize_answer(chosen)

    # --- rejected: on-policy verbose or invalid, non-empty, != chosen ---
    bad = [
        d
        for d in labeled
        if d["norm"] and d["norm"] != chosen_norm and ((not d["valid"]) or d["wc"] >= verbose_min_words)
    ]
    if not bad:
        return None  # skip: no negative to push down (all samples short+valid or == chosen)

    # prefer the actual top-1 sample if it is bad (that's literally the failure we correct).
    top1 = labeled[0] if labeled else None
    if top1 is not None and top1 in bad:
        rejected_d = top1
    else:
        rejected_d = next(d for d in bad if d["raw"] == _most_frequent([d2["raw"] for d2 in bad]))

    rejected = rejected_d["raw"]
    rejected_source = "on_policy_verbose" if rejected_d["wc"] >= verbose_min_words else "on_policy_invalid"
    return {
        "pair_id": f"{input_a}+{input_b}",
        "input_a": input_a,
        "input_b": input_b,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_source": chosen_source,
        "rejected_source": rejected_source,
        "chosen_len": _word_count(chosen),
        "rejected_len": rejected_d["wc"],
        "chosen_valid": True,
        "rejected_valid": bool(rejected_d["valid"]),
    }


# --------------------------------------------------------------------------- #
# Generation + orchestration (GPU) — kept out of the pure logic above.
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="SFT run_dir (soft_ce) with config.yaml + adapter.")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--input_file", required=True, help="train.jsonl subset with candidate_outputs.")
    parser.add_argument("--output_dir", required=True, help="Where to write pairs.jsonl + pairs.meta.json.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--strip_think", type=lambda s: s.lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--close_think_prompt", type=lambda s: s.lower() in {"1", "true", "yes"}, default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import torch  # heavy imports are local so the pure logic stays importable without them.
    from peft import PeftModel

    from src.eval.run_sft_eval import (
        decode_generated_text,
        load_eval_records,
        render_generation_prompt,
        resolve_adapter_dir,
        strip_think_block,
    )
    from src.sft.config import SFTConfig
    from src.sft.trainer import build_model_and_tokenizer

    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    config = SFTConfig(**json.loads(json.dumps(_load_yaml(run_dir / "config.yaml"))))

    model, tokenizer = build_model_and_tokenizer(config, apply_lora=False)
    adapter_dir = resolve_adapter_dir(run_dir, args.adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    records = load_eval_records(args.input_file, max_examples=args.max_examples)
    # keep candidate rows (rank/source) for the fallback chosen.
    raw_rows = _load_candidate_rows(args.input_file, args.max_examples)

    pairs: list[dict[str, Any]] = []
    skips = Counter()
    for record in records:
        prompt = render_generation_prompt(
            record, config, tokenizer, close_think=args.close_think_prompt
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            sequences = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = decode_generated_text(tokenizer, sequences, inputs["input_ids"].shape[1])
        raw_samples = [strip_think_block(t).strip() if args.strip_think else t.strip() for t in decoded]

        candidate_rows = raw_rows.get(record.pair_id, [])
        pair = select_preference_pair(
            record.input_a,
            record.input_b,
            record.known_outputs,
            record.canonical_output,
            raw_samples,
            candidate_rows=candidate_rows,
        )
        if pair is None:
            skips["skipped"] += 1
            continue
        pairs.append(pair)

    output_dir = Path(args.output_dir)
    _write_jsonl(output_dir / "pairs.jsonl", pairs)
    meta = {
        "model": config.model_name_or_path,
        "adapter": str(adapter_dir),
        "prompt_format": config.prompt_format,
        "system_prompt": config.system_prompt,
        "max_seq_length": args.max_seq_length,
        "tokenizer": config.model_name_or_path,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "n_recipes": len(records),
        "n_pairs": len(pairs),
        "n_skipped": int(skips["skipped"]),
    }
    (output_dir / "pairs.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[dpo-pairs] {len(pairs)} pairs from {len(records)} recipes ({skips['skipped']} skipped) -> {output_dir}")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_candidate_rows(path: str, max_examples: int | None) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            if max_examples is not None and len(rows) >= max_examples:
                break
            rec = json.loads(line)
            pair_id = str(rec.get("pair_id") or f"{rec['input_a']}+{rec['input_b']}")
            rows[pair_id] = rec.get("candidate_outputs", []) or []
    return rows


if __name__ == "__main__":
    main()
