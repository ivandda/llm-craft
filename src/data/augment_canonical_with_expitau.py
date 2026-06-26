"""Augment recipe_canonical_v0.jsonl with matching expitau observations.

Behavior:
- Loads the canonical file (reasonable size) into memory preserving order.
- Streams recipe_observations_expitau.jsonl (large) line-by-line.
- For any observation where normalized (input_a,input_b) matches a canonical pair,
  appends the observation.output to that record's `known_outputs` if not already present.
- Writes an augmented jsonl to output path.

Usage:
  python src/data/augment_canonical_with_expitau.py \
      --canonical datasets/processed/recipe_canonical_v0.jsonl \
      --observations datasets/processed/recipe_observations_expitau.jsonl \
      --output datasets/processed/recipe_canonical_v0.expitau_augmented.jsonl

Only the `known_outputs` field is modified.
"""

import json
import argparse
import os
from typing import Tuple, List, Optional
from src.data.normalize import parse_expitau, get_emoji
from src.data.schemas import RecipeObservation


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(s.strip().lower().split())


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return (normalize_text(a), normalize_text(b))


def create_expitau_observations(base_dir: str, out_path: str):
    """Parse raw expitau data, canonicalize, enrich with emojis, and write to out_path.

    Overwrites out_path if it exists.
    """
    raw = parse_expitau(base_dir)
    processed: List[RecipeObservation] = []

    for input_a, input_b, output, emoji_out, source in raw:
        if input_a.strip().lower() <= input_b.strip().lower():
            sorted_a, sorted_b = input_a.strip(), input_b.strip()
        else:
            sorted_a, sorted_b = input_b.strip(), input_a.strip()

        emoji_a = get_emoji(sorted_a)
        emoji_b = get_emoji(sorted_b)
        final_emoji_out = emoji_out or get_emoji(output)

        obs = RecipeObservation(
            input_a=sorted_a,
            input_b=sorted_b,
            output=output.strip(),
            emoji_a=emoji_a,
            emoji_b=emoji_b,
            emoji_output=final_emoji_out,
            source=source,
        )
        processed.append(obs)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for obs in processed:
            fout.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")

    print(f"Wrote {len(processed)} expitau observations to {out_path}")


def augment(canonical_path: str, observations_path: str, output_path: str):
    if not os.path.exists(canonical_path):
        raise FileNotFoundError(f"Canonical file not found: {canonical_path}")

    # Create or refresh the expitau observations file first so augmentation uses up-to-date data.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    create_expitau_observations(base_dir, observations_path)

    # Load canonical into memory (keeps order)
    records = []
    mapping = {}  # (input_a_norm, input_b_norm) -> record
    with open(canonical_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # keep malformed lines as-is by wrapping in a dict
                continue
            records.append(obj)
            key = pair_key(obj.get("input_a", ""), obj.get("input_b", ""))
            mapping[key] = obj

    matched_obs = 0
    added_outputs = 0
    seen_pairs = set()

    # Stream observations file
    with open(observations_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # Only consider records from expitau source (robust substring check)
            src = obj.get("source", "") or ""
            if "expitau" not in src.lower():
                continue

            key = pair_key(obj.get("input_a", ""), obj.get("input_b", ""))
            rec = mapping.get(key)
            if rec is None:
                continue

            matched_obs += 1
            out = obj.get("output")
            if not out:
                continue

            known = rec.get("known_outputs")
            if known is None:
                known = []
                rec["known_outputs"] = known

            # Append only if not present (exact string match). Normalize not applied to outputs to preserve original text.
            if out not in known:
                known.append(out)
                added_outputs += 1
                seen_pairs.add(key)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write augmented canonical file preserving original order
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Augmentation complete.")
    print(f"Observation lines matched to canonical pairs: {matched_obs}")
    print(f"New outputs added to known_outputs: {added_outputs}")
    print(f"Canonical pairs updated: {len(seen_pairs)}")


def parse_args():
    p = argparse.ArgumentParser(description="Augment canonical recipes with expitau observations")
    p.add_argument("--canonical", default="datasets/processed/recipe_canonical_v0.jsonl", help="path to canonical jsonl")
    p.add_argument("--observations", default="datasets/processed/recipe_observations_expitau.jsonl", help="path to expitau observations jsonl")
    p.add_argument("--output", default="datasets/processed/recipe_canonical_v0.expitau_augmented.jsonl", help="output augmented canonical jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment(args.canonical, args.observations, args.output)
