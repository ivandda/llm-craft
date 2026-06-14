import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Any

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_canonical_v0.jsonl")
    processed_dir = os.path.join(base_dir, "datasets", "processed")

    if not os.path.exists(input_file):
        print(f"Error: Canonical file not found at {input_file}. Run clean.py first.")
        return

    # Seed random for deterministic sampling
    random.seed(42)

    # Reservoirs for each target evaluation dataset
    reservoirs: Dict[str, List[Dict[str, Any]]] = {
        "dev_keep": [],          # Target: 1000
        "test_keep": [],         # Target: 1000
        "test_identity": [],     # Target: 500
        "test_conflicting": []   # Target: 500
    }
    
    limits = {
        "dev_keep": 1000,
        "test_keep": 1000,
        "test_identity": 500,
        "test_conflicting": 500
    }

    counts = {
        "dev_keep": 0,
        "test_keep": 0,
        "test_identity": 0,
        "test_conflicting": 0
    }

    print("Pass 1: Running reservoir sampling over canonical recipes (streaming)...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                recipe = json.loads(line)
            except Exception:
                continue

            split = recipe.get("split")
            status = recipe.get("status")

            # Match recipe to one of our target evaluation categories
            category = None
            if split == "dev" and status == "keep":
                category = "dev_keep"
            elif split == "test":
                if status == "keep":
                    category = "test_keep"
                elif status == "review_identity":
                    category = "test_identity"
                elif status == "keep_conflicting":
                    category = "test_conflicting"

            if category is None:
                continue

            counts[category] += 1
            seen_count = counts[category]
            limit = limits[category]

            # Reservoir sampling update logic
            if len(reservoirs[category]) < limit:
                reservoirs[category].append(recipe)
            else:
                r = random.randint(0, seen_count - 1)
                if r < limit:
                    reservoirs[category][r] = recipe

    # Extract all pair_ids that were sampled for Pass 2
    sampled_pair_ids = set()
    for cat, items in reservoirs.items():
        for item in items:
            sampled_pair_ids.add(item["pair_id"])

    print(f"Sampled {len(sampled_pair_ids)} unique pairs. Pass 2: Collecting all known outputs for these pairs...")
    
    # Map pair_id -> set of outputs to verify multiple alternative correct answers
    pair_outputs = defaultdict(set)
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                recipe = json.loads(line)
            except Exception:
                continue
            
            pair_id = recipe.get("pair_id")
            if pair_id in sampled_pair_ids:
                pair_outputs[pair_id].add(recipe.get("output"))

    print("Formatting and writing evaluation sets...")
    
    eval_sets = {
        "eval_dev_1k.jsonl": reservoirs["dev_keep"],
        "eval_test_1k.jsonl": reservoirs["test_keep"],
        "eval_test_identity_500.jsonl": reservoirs["test_identity"],
        "eval_test_conflicting_500.jsonl": reservoirs["test_conflicting"]
    }

    for filename, recipes in eval_sets.items():
        path = os.path.join(processed_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for recipe in recipes:
                pair_id = recipe["pair_id"]
                eval_item = {
                    "pair_id": pair_id,
                    "pair_key": recipe["pair_key"],
                    "input_a": recipe["input_a"],
                    "input_b": recipe["input_b"],
                    "known_outputs": sorted(list(pair_outputs[pair_id])),
                    "canonical_output": recipe["output"],
                    "status": recipe["status"],
                    "split": recipe["split"]
                }
                f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
        print(f"  Exported {len(recipes)} items to {path}")

if __name__ == "__main__":
    main()
