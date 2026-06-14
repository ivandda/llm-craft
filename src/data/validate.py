import os
import json
from collections import defaultdict

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_observations_v0.jsonl")

    if not os.path.exists(input_file):
        print(f"Error: Processed file not found at {input_file}. Please run normalization first.")
        return

    num_observations = 0
    parse_errors = 0
    empty_output_count = 0
    
    unique_pairs = set()
    unique_recipes = set()
    sources = set()
    pair_to_outputs = defaultdict(set)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            num_observations += 1
            if not line.strip():
                continue
            try:
                obs = json.loads(line)
            except Exception:
                parse_errors += 1
                continue

            input_a = obs.get("input_a", "")
            input_b = obs.get("input_b", "")
            output = obs.get("output", "")
            source = obs.get("source", "")

            if not output or not output.strip():
                empty_output_count += 1

            # Case-insensitive keys for unique counts and conflicts
            key_a = input_a.strip().lower()
            key_b = input_b.strip().lower()
            key_out = output.strip().lower()

            # Ensure alphabetical sorting for the pair key
            pair_key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)
            recipe_key = (pair_key[0], pair_key[1], key_out)

            unique_pairs.add(pair_key)
            unique_recipes.add(recipe_key)
            if source:
                sources.add(source)
            pair_to_outputs[pair_key].add(key_out)

    num_unique_pairs = len(unique_pairs)
    num_unique_recipes = len(unique_recipes)
    num_sources = len(sources)
    parse_error_rate = parse_errors / num_observations if num_observations > 0 else 0.0
    duplicate_recipe_count = num_observations - num_unique_recipes - parse_errors
    conflicting_pair_count = sum(1 for outputs in pair_to_outputs.values() if len(outputs) > 1)

    report = {
        "num_observations": num_observations,
        "num_unique_pairs": num_unique_pairs,
        "num_unique_recipes": num_unique_recipes,
        "num_sources": num_sources,
        "parse_error_rate": parse_error_rate,
        "empty_output_count": empty_output_count,
        "duplicate_recipe_count": duplicate_recipe_count,
        "conflicting_pair_count": conflicting_pair_count
    }

    print("\nDataset Validation Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
