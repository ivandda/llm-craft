import os
import json

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_canonical_v0.jsonl")
    
    output_files = {
        "train": os.path.join(base_dir, "datasets", "processed", "sft_train.jsonl"),
        "dev": os.path.join(base_dir, "datasets", "processed", "sft_dev.jsonl"),
        "test": os.path.join(base_dir, "datasets", "processed", "sft_test.jsonl")
    }

    if not os.path.exists(input_file):
        print(f"Error: Canonical file not found at {input_file}. Run clean.py first.")
        return

    print("Exporting SFT baseline datasets...")
    
    file_handles = {split: open(path, "w", encoding="utf-8") for split, path in output_files.items()}
    counts = {"train": 0, "dev": 0, "test": 0}

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                recipe = json.loads(line)
            except Exception:
                continue

            status = recipe.get("status", "")
            # Only export recipes that are kept or set for review (dropping explicit structural errors)
            if status == "drop_empty_output":
                continue

            split = recipe.get("split", "")
            if split not in file_handles:
                continue

            input_a = recipe.get("input_a", "")
            input_b = recipe.get("input_b", "")
            output = recipe.get("output", "")

            # Format in standard conversational format (messages)
            sft_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Combine the concepts: {input_a} + {input_b}. Return only the resulting concept."
                    },
                    {
                        "role": "assistant",
                        "content": output
                    }
                ],
                "metadata": {
                    "pair_key": recipe.get("pair_key"),
                    "recipe_key": recipe.get("recipe_key"),
                    "source_count": recipe.get("source_count"),
                    "is_conflicting_pair": recipe.get("is_conflicting_pair"),
                    "split": split
                }
            }

            file_handles[split].write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            counts[split] += 1

    # Close all files
    for fh in file_handles.values():
        fh.close()

    print("\nSFT Export complete:")
    for split, count in counts.items():
        print(f"  {split}: {count} recipes -> {output_files[split]}")

if __name__ == "__main__":
    main()
