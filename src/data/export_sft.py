import os
import json
import yaml
from typing import Dict, List, TextIO

def get_sft_prompt(template: str, input_a: str, input_b: str) -> str:
    return template.format(input_a=input_a, input_b=input_b)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_file = os.path.join(base_dir, "configs", "pipeline_config.yaml")
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_canonical_v0.jsonl")
    processed_dir = os.path.join(base_dir, "datasets", "processed")

    # Load configuration
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "sft_export": {
                "exclude_conflicts": False,
                "exclude_identities": True,
                "prompt_template": "Given two concepts, combine them into one resulting concept.\n\nConcept A: {input_a}\nConcept B: {input_b}\n\nReturn only the resulting concept."
            }
        }

    sft_cfg = config.get("sft_export", {})
    exclude_conflicts = sft_cfg.get("exclude_conflicts", False)
    exclude_identities = sft_cfg.get("exclude_identities", True)
    prompt_template = sft_cfg.get("prompt_template", "Given two concepts, combine them into one resulting concept.\n\nConcept A: {input_a}\nConcept B: {input_b}\n\nReturn only the resulting concept.")

    variants = sft_cfg.get("variants", ["clean", "all"])
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

            split = recipe.get("split", "")
            if split not in splits:
                continue

            input_a = recipe.get("input_a", "")
            input_b = recipe.get("input_b", "")
            status = recipe.get("status", "")
            is_conflicting = recipe.get("is_conflicting_pair", False)

            # Metadata shared template
            metadata_template = {
                "pair_id": recipe.get("pair_id"),
                "pair_key": recipe.get("pair_key"),
                "source_count": recipe.get("source_count"),
                "pair_num_outputs": recipe.get("pair_num_outputs"),
                "is_conflicting_pair": is_conflicting,
                "status": status,
                "split": split
            }

            # 1. Export CLEAN variant (strictly non-identity, and non-conflicting if config requests it)
            if exclude_identities and status == "review_identity":
                pass
            elif exclude_conflicts and is_conflicting:
                pass
            else:
                canonical_output = recipe.get("canonical_output", "")
                
                # Fetch output recipe ID (deterministic hash for the single output)
                # Compute stable recipe ID
                # We can construct recipe_id as sha256 of input_a, input_b, canonical_output
                from src.data.schemas import stable_id
                recipe_id = stable_id([input_a, input_b, canonical_output])

                clean_metadata = dict(metadata_template)
                clean_metadata["recipe_id"] = recipe_id
                clean_metadata["recipe_key"] = f"{recipe.get('pair_key')}=>{canonical_output}"
                clean_metadata["observation_count"] = recipe.get("observation_counts")[recipe.get("known_outputs").index(canonical_output)]

                sft_item_clean = {
                    "messages": [
                        {
                            "role": "user",
                            "content": get_sft_prompt(prompt_template, input_a, input_b)
                        },
                        {
                            "role": "assistant",
                            "content": canonical_output
                        }
                    ],
                    "metadata": clean_metadata
                }
                if "clean" in variants:
                    file_handles["clean"][split].write(json.dumps(sft_item_clean, ensure_ascii=False) + "\n")
                    counts["clean"][split] += 1

            # 2. Export ALL variant (preserves ALL creative outputs)
            if "all" in variants:
                known_outputs = recipe.get("known_outputs", [])
                known_emojis = recipe.get("known_emojis", [])
                observation_counts = recipe.get("observation_counts", [])

                for i, output in enumerate(known_outputs):
                    # We can also check if we want to exclude identities in sft_all if config requested it
                    # Usually sft_all keeps everything, but let's respect exclude_identities if it's set
                    # In standard SFT training, keeping identity copies in "sft_all" is normal, but let's follow the status
                    is_identity_output = (output == input_a or output == input_b)
                    if exclude_identities and is_identity_output:
                        continue

                    from src.data.schemas import stable_id
                    recipe_id = stable_id([input_a, input_b, output])

                    all_metadata = dict(metadata_template)
                    all_metadata["recipe_id"] = recipe_id
                    all_metadata["recipe_key"] = f"{recipe.get('pair_key')}=>{output}"
                    all_metadata["observation_count"] = observation_counts[i]

                    sft_item_all = {
                        "messages": [
                            {
                                "role": "user",
                                "content": get_sft_prompt(prompt_template, input_a, input_b)
                            },
                            {
                                "role": "assistant",
                                "content": output
                            }
                        ],
                        "metadata": all_metadata
                    }
                    file_handles["all"][split].write(json.dumps(sft_item_all, ensure_ascii=False) + "\n")
                    counts["all"][split] += 1

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
