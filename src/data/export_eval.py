import os
import json
import yaml
import random
from typing import Dict, List, Any

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_file = os.path.join(base_dir, "configs", "pipeline_config.yaml")
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_canonical_v0.jsonl")
    processed_dir = os.path.join(base_dir, "datasets", "processed")

    if not os.path.exists(input_file):
        print(f"Error: Canonical file not found at {input_file}. Run clean.py first.")
        return

    # Load configuration
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "evaluation_export": {
                "sizes": {
                    "dev_keep": 1000,
                    "test_keep": 1000,
                    "test_identity": 500,
                    "test_conflicting": 500
                }
            }
        }

    eval_cfg = config.get("evaluation_export", {})
    sizes = eval_cfg.get("sizes", {"dev_keep": 1000, "test_keep": 1000, "test_identity": 500, "test_conflicting": 500})

    # Seed random for deterministic sampling
    random.seed(42)

    # Reservoirs for each target evaluation dataset
    reservoirs: Dict[str, List[Dict[str, Any]]] = {
        "dev_keep": [],
        "test_keep": [],
        "test_identity": [],
        "test_conflicting": []
    }
    
    limits = {
        "dev_keep": sizes.get("dev_keep", 1000),
        "test_keep": sizes.get("test_keep", 1000),
        "test_identity": sizes.get("test_identity", 500),
        "test_conflicting": sizes.get("test_conflicting", 500)
    }

    counts = {
        "dev_keep": 0,
        "test_keep": 0,
        "test_identity": 0,
        "test_conflicting": 0
    }

    print("Running reservoir sampling over canonical recipes (single pass)...")
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

            category = None
            if split == "dev" and status == "keep":
                category = "dev_keep"
            elif split == "test":
                if status == "review_identity":
                    category = "test_identity"
                elif recipe.get("is_conflicting_pair", False):
                    category = "test_conflicting"
                elif status == "keep":
                    category = "test_keep"

            if category is None:
                continue

            counts[category] += 1
            seen_count = counts[category]
            limit = limits[category]

            if len(reservoirs[category]) < limit:
                reservoirs[category].append(recipe)
            else:
                r = random.randint(0, seen_count - 1)
                if r < limit:
                    reservoirs[category][r] = recipe

    print("Writing evaluation sets...")
    def format_size(val: int) -> str:
        if val >= 1000 and val % 1000 == 0:
            return f"{val // 1000}k"
        return str(val)

    dev_size_str = format_size(limits["dev_keep"])
    test_size_str = format_size(limits["test_keep"])
    identity_size_str = format_size(limits["test_identity"])
    conflicting_size_str = format_size(limits["test_conflicting"])

    eval_sets = {
        f"eval_dev_{dev_size_str}.jsonl": reservoirs["dev_keep"],
        f"eval_test_{test_size_str}.jsonl": reservoirs["test_keep"],
        f"eval_test_identity_{identity_size_str}.jsonl": reservoirs["test_identity"],
        f"eval_test_conflicting_{conflicting_size_str}.jsonl": reservoirs["test_conflicting"]
    }

    for filename, recipes in eval_sets.items():
        path = os.path.join(processed_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for recipe in recipes:
                eval_item = {
                    "pair_id": recipe["pair_id"],
                    "pair_key": recipe["pair_key"],
                    "input_a": recipe["input_a"],
                    "input_b": recipe["input_b"],
                    "known_outputs": recipe["known_outputs"],
                    "canonical_output": recipe["canonical_output"],
                    "status": recipe["status"],
                    "split": recipe["split"]
                }
                f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
        print(f"  Exported {len(recipes)} items to {path}")

if __name__ == "__main__":
    main()
