import os
import json
import yaml
from typing import Dict, List, TextIO

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_file = os.path.join(base_dir, "configs", "pipeline_config.yaml")
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_canonical_v0.jsonl")
    processed_dir = os.path.join(base_dir, "datasets", "processed")

    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {"recipe_export": {"exclude_identities": True}}

    recipe_cfg = config.get("recipe_export", config.get("sft_export", {}))
    exclude_identities = recipe_cfg.get("exclude_identities", True)

    splits = ["train", "dev", "test"]

    if not os.path.exists(input_file):
        print(f"Error: Canonical file not found at {input_file}. Run clean.py first.")
        return

    file_handles: Dict[str, TextIO] = {}
    counts: Dict[str, int] = {}
    for split in splits:
        path = os.path.join(processed_dir, f"recipes_{split}.jsonl")
        file_handles[split] = open(path, "w", encoding="utf-8")
        counts[split] = 0

    print("Exporting minimal recipe split files...")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                recipe = json.loads(line)
            except Exception:
                continue

            split = recipe.get("split", "")
            if split not in splits:
                continue

            input_a = recipe.get("input_a", "")
            input_b = recipe.get("input_b", "")
            status = recipe.get("status", "")
            if exclude_identities and status == "review_identity":
                continue

            outputs = recipe.get("known_outputs", [])
            if not outputs:
                continue

            recipe_item = {
                "input_a": input_a,
                "input_b": input_b,
                "outputs": outputs,
            }
            file_handles[split].write(json.dumps(recipe_item, ensure_ascii=False) + "\n")
            counts[split] += 1

    for handle in file_handles.values():
        handle.close()

    print("\nRecipe export complete:")
    for split in splits:
        path = os.path.join(processed_dir, f"recipes_{split}.jsonl")
        print(f"  {split}: {counts[split]} pairs -> {path}")

if __name__ == "__main__":
    main()
