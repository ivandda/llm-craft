import os
import json
import hashlib
from collections import defaultdict
from typing import Tuple, Set, Dict
from src.data.schemas import stable_id

def stable_bucket(value: Tuple[str, str], modulo: int = 10_000) -> int:
    str_val = f"{value[0]}+{value[1]}"
    digest = hashlib.sha256(str_val.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo

def assign_split(pair_key: Tuple[str, str]) -> str:
    bucket = stable_bucket(pair_key)
    if bucket < 8000:
        return "train"
    elif bucket < 9000:
        return "dev"
    else:
        return "test"

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_observations_v0.jsonl")
    output_file = os.path.join(base_dir, "datasets", "processed", "recipe_canonical_v0.jsonl")
    report_file = os.path.join(base_dir, "datasets", "reports", "clean_metrics.json")

    if not os.path.exists(input_file):
        print(f"Error: Input observations file not found at {input_file}")
        return

    print("Pass 1: Analyzing casing frequencies, emojis, conflicts, and sources...")
    
    # Frequency trackers for mode casing resolution
    casing_counts = defaultdict(lambda: defaultdict(int))
    emoji_counts = defaultdict(lambda: defaultdict(int))
    
    # Recipe properties
    pair_outputs = defaultdict(set)
    recipe_sources = defaultdict(set)
    recipe_obs_count = defaultdict(int)

    line_count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"  Processed {line_count} lines...")

            try:
                obs = json.loads(line)
            except Exception:
                continue

            input_a = obs.get("input_a", "").strip()
            input_b = obs.get("input_b", "").strip()
            output = obs.get("output", "").strip()
            emoji_a = obs.get("emoji_a")
            emoji_b = obs.get("emoji_b")
            emoji_output = obs.get("emoji_output")
            source = obs.get("source", "")

            # Normalized keys
            key_a = input_a.lower()
            key_b = input_b.lower()
            key_out = output.lower()

            pair_key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
            recipe_key = (pair_key[0], pair_key[1], key_out)

            # Record casing counts
            if input_a:
                casing_counts[key_a][input_a] += 1
            if input_b:
                casing_counts[key_b][input_b] += 1
            if output:
                casing_counts[key_out][output] += 1

            # Record emoji counts
            if emoji_a and emoji_a != "⚪":
                emoji_counts[key_a][emoji_a] += 1
            if emoji_b and emoji_b != "⚪":
                emoji_counts[key_b][emoji_b] += 1
            if emoji_output and emoji_output != "⚪":
                emoji_counts[key_out][emoji_output] += 1

            # Record outputs, sources, and counts
            pair_outputs[pair_key].add(key_out)
            if source:
                recipe_sources[recipe_key].add(source)
            recipe_obs_count[recipe_key] += 1

    print("Resolving casings and emojis...")
    resolved_casing = {}
    for norm_name, casings in casing_counts.items():
        # Choose the casing that appears most frequently
        resolved_casing[norm_name] = max(casings.items(), key=lambda x: x[1])[0]

    resolved_emoji = {}
    for norm_name, emojis in emoji_counts.items():
        if emojis:
            # Choose the emoji that appears most frequently
            resolved_emoji[norm_name] = max(emojis.items(), key=lambda x: x[1])[0]
        else:
            resolved_emoji[norm_name] = "⚪"

    print("Pass 2: Generating canonical recipes, deduplicating, and assigning splits...")
    
    written_recipes = set()
    metrics = {
        "num_canonical_recipes": 0,
        "num_unique_pairs": len(pair_outputs),
        "num_conflicting_pairs": 0,
        "num_keep": 0,
        "num_keep_conflicting": 0,
        "num_review_identity": 0,
        "num_drop_empty_output": 0,
        "split_counts": {"train": 0, "dev": 0, "test": 0}
    }

    # Count conflicting pairs
    for p_key, outputs in pair_outputs.items():
        if len(outputs) > 1:
            metrics["num_conflicting_pairs"] += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    line_count = 0
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"  Processed {line_count} lines...")

            try:
                obs = json.loads(line)
            except Exception:
                continue

            input_a = obs.get("input_a", "").strip()
            input_b = obs.get("input_b", "").strip()
            output = obs.get("output", "").strip()

            key_a = input_a.lower()
            key_b = input_b.lower()
            key_out = output.lower()

            pair_key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
            recipe_key = (pair_key[0], pair_key[1], key_out)

            if recipe_key in written_recipes:
                continue
            written_recipes.add(recipe_key)

            # Resolve casing
            display_a = resolved_casing.get(pair_key[0], input_a)
            display_b = resolved_casing.get(pair_key[1], input_b)
            display_out = resolved_casing.get(key_out, output)

            # Resolve emoji
            emoji_a = resolved_emoji.get(pair_key[0], "⚪")
            emoji_b = resolved_emoji.get(pair_key[1], "⚪")
            emoji_out = resolved_emoji.get(key_out, "⚪")

            # Stats
            obs_count = recipe_obs_count[recipe_key]
            sources = sorted(list(recipe_sources[recipe_key]))
            pair_num_outputs = len(pair_outputs[pair_key])
            is_conflicting = pair_num_outputs > 1

            # Split assignment (hashing of pair_key ensures no leakage)
            split = assign_split(pair_key)

            # Status assignment
            if not display_out:
                status = "drop_empty_output"
                metrics["num_drop_empty_output"] += 1
            elif key_out == key_a or key_out == key_b:
                status = "review_identity"
                metrics["num_review_identity"] += 1
            elif is_conflicting:
                status = "keep_conflicting"
                metrics["num_keep_conflicting"] += 1
                metrics["num_keep"] += 1
            else:
                status = "keep"
                metrics["num_keep"] += 1

            metrics["num_canonical_recipes"] += 1
            metrics["split_counts"][split] += 1

            pair_id = stable_id([pair_key[0], pair_key[1]])
            recipe_id = stable_id([pair_key[0], pair_key[1], key_out])

            canonical_recipe = {
                "pair_id": pair_id,
                "recipe_id": recipe_id,
                "pair_key": f"{pair_key[0]}+{pair_key[1]}",
                "recipe_key": f"{pair_key[0]}+{pair_key[1]}=>{key_out}",
                "input_a": display_a,
                "input_b": display_b,
                "output": display_out,
                "emoji_a": emoji_a,
                "emoji_b": emoji_b,
                "emoji_output": emoji_out,
                "sources": sources,
                "source_count": len(sources),
                "observation_count": obs_count,
                "pair_num_outputs": pair_num_outputs,
                "is_conflicting_pair": is_conflicting,
                "split": split,
                "status": status
            }

            fout.write(json.dumps(canonical_recipe, ensure_ascii=False) + "\n")

    # Write report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as frep:
        frep.write(json.dumps(metrics, indent=2) + "\n")

    print("\nData cleaning and splits assigned successfully!")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
