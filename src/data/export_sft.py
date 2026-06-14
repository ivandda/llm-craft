import os
import json
from typing import Dict, List, TextIO

def get_sft_prompt(input_a: str, input_b: str) -> str:
    return (
        "Given two concepts, combine them into one resulting concept.\n\n"
        f"Concept A: {input_a}\n"
        f"Concept B: {input_b}\n\n"
        "Return only the resulting concept."
    )

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_canonical_v0.jsonl")
    
    # SFT outputs directory
    processed_dir = os.path.join(base_dir, "datasets", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # SFT Variants:
    # 1. sft_clean: Only keep recipes with status 'keep' and is_conflicting_pair=False (high confidence, non-identity, non-ambiguous)
    # 2. sft_all: All parsed valid recipes (status in ['keep', 'keep_conflicting', 'review_identity'])
    variants = ["clean", "all"]
    splits = ["train", "dev", "test"]
    
    file_handles: Dict[str, Dict[str, TextIO]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    
    for var in variants:
        file_handles[var] = {}
        counts[var] = {}
        for split in splits:
            path = os.path.join(processed_dir, f"sft_{var}_{split}.jsonl")
            file_handles[var][split] = open(path, "w", encoding="utf-8")
            counts[var][split] = 0

    if not os.path.exists(input_file):
        print(f"Error: Canonical file not found at {input_file}. Run clean.py first.")
        return

    print("Exporting SFT dataset variants...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                recipe = json.loads(line)
            except Exception:
                continue

            status = recipe.get("status", "")
            if status == "drop_empty_output":
                continue

            split = recipe.get("split", "")
            if split not in splits:
                continue

            input_a = recipe.get("input_a", "")
            input_b = recipe.get("input_b", "")
            output = recipe.get("output", "")
            is_conflicting = recipe.get("is_conflicting_pair", False)

            # Build metadata matching all requirements
            metadata = {
                "pair_id": recipe.get("pair_id"),
                "recipe_id": recipe.get("recipe_id"),
                "pair_key": recipe.get("pair_key"),
                "recipe_key": recipe.get("recipe_key"),
                "source_count": recipe.get("source_count"),
                "observation_count": recipe.get("observation_count"),
                "pair_num_outputs": recipe.get("pair_num_outputs"),
                "is_conflicting_pair": is_conflicting,
                "status": status,
                "split": split
            }

            # Standardized SFT message format
            sft_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": get_sft_prompt(input_a, input_b)
                    },
                    {
                        "role": "assistant",
                        "content": output
                    }
                ],
                "metadata": metadata
            }

            sft_line = json.dumps(sft_item, ensure_ascii=False) + "\n"

            # Route to appropriate variants
            # Variant: ALL
            if status in ["keep", "keep_conflicting", "review_identity"]:
                file_handles["all"][split].write(sft_line)
                counts["all"][split] += 1

            # Variant: CLEAN
            if status == "keep" and not is_conflicting:
                file_handles["clean"][split].write(sft_line)
                counts["clean"][split] += 1

    # Close handles
    for var in variants:
        for split in splits:
            file_handles[var][split].close()

    print("\nSFT Export complete:")
    for var in variants:
        print(f"Variant: {var.upper()}")
        for split in splits:
            path = os.path.join(processed_dir, f"sft_{var}_{split}.jsonl")
            print(f"  {split}: {counts[var][split]} recipes -> {path}")

if __name__ == "__main__":
    main()
