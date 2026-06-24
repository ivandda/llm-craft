import os
import json
import yaml
import random
from typing import Dict, List, Any


ALL_SIZE = "all"


def parse_size(value: Any, key: str) -> int | str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == ALL_SIZE:
            return ALL_SIZE
        raise ValueError(f"Invalid evaluation_export.sizes.{key}: {value!r}. Use a positive integer or 'all'.")
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(f"Invalid evaluation_export.sizes.{key}: {value!r}. Use a positive integer or 'all'.")


def should_keep_record(reservoir: list[dict[str, Any]], limit: int | str, seen_count: int) -> int | None:
    if limit == ALL_SIZE:
        return len(reservoir)
    if len(reservoir) < limit:
        return len(reservoir)
    r = random.randint(0, seen_count - 1)
    if r < limit:
        return r
    return None


def format_size(value: int | str) -> str:
    if value == ALL_SIZE:
        return ALL_SIZE
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)

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
                    "test_keep": 1000
                }
            }
        }

    eval_cfg = config.get("evaluation_export", {})
    sizes = eval_cfg.get("sizes", {"dev_keep": 1000, "test_keep": 1000})
    include_pair_id = eval_cfg.get("include_pair_id", False)
    include_canonical_output = eval_cfg.get("include_canonical_output", False)

    # Seed random for deterministic sampling
    random.seed(42)

    # Reservoirs for each target evaluation dataset
    reservoirs: Dict[str, List[Dict[str, Any]]] = {
        "dev_keep": [],
        "test_keep": []
    }
    
    limits: Dict[str, int | str] = {
        "dev_keep": parse_size(sizes.get("dev_keep", 1000), "dev_keep"),
        "test_keep": parse_size(sizes.get("test_keep", 1000), "test_keep")
    }

    counts = {
        "dev_keep": 0,
        "test_keep": 0
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
            elif split == "test" and status == "keep":
                category = "test_keep"

            if category is None:
                continue

            counts[category] += 1
            seen_count = counts[category]
            limit = limits[category]

            write_index = should_keep_record(reservoirs[category], limit, seen_count)
            if write_index is None:
                continue
            if write_index == len(reservoirs[category]):
                reservoirs[category].append(recipe)
            else:
                reservoirs[category][write_index] = recipe

    print("Writing evaluation sets...")
    dev_size_str = format_size(limits["dev_keep"])
    test_size_str = format_size(limits["test_keep"])

    eval_sets = {
        f"eval_dev_{dev_size_str}.jsonl": reservoirs["dev_keep"],
        f"eval_test_{test_size_str}.jsonl": reservoirs["test_keep"]
    }

    for filename, recipes in eval_sets.items():
        path = os.path.join(processed_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for recipe in recipes:
                eval_item = {
                    "input_a": recipe["input_a"],
                    "input_b": recipe["input_b"],
                    "known_outputs": recipe["known_outputs"]
                }
                if include_pair_id:
                    eval_item["pair_id"] = recipe["pair_id"]
                if include_canonical_output:
                    eval_item["canonical_output"] = recipe["canonical_output"]
                f.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
        print(f"  Exported {len(recipes)} items to {path}")

if __name__ == "__main__":
    main()
