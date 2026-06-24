import os
import json
import yaml
from collections import defaultdict
from typing import Tuple, Set, Dict, List
from src.data.schemas import stable_id, RecipeObservation

def stable_bucket(value: Tuple[str, str], modulo: int = 10_000) -> int:
    str_val = f"{value[0]}+{value[1]}"
    # Hash of the sorted pair key ensures consistency across splits
    digest = hashlib.sha256(str_val.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo

# We can import hashlib inside the script
import hashlib
from collections import Counter
from src.data.quality import normalize_concept, quality_config_from_dict, recipe_quality_reason

def assign_split(pair_key: Tuple[str, str], split_ratios: Dict[str, float]) -> str:
    # Modulo-based splitting using pair keys
    str_val = f"{pair_key[0]}+{pair_key[1]}"
    digest = hashlib.sha256(str_val.encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) % 10000

    train_limit = int(split_ratios.get("train", 0.80) * 10000)
    dev_limit = train_limit + int(split_ratios.get("dev", 0.10) * 10000)

    if bucket < train_limit:
        return "train"
    elif bucket < dev_limit:
        return "dev"
    else:
        return "test"

def is_garbage_concept(text: str) -> bool:
    if not text:
        return True
    text = text.strip()
    # 1. Contains hashtags, backticks, or other code/formatting symbols
    if any(char in text for char in ("#", "`", ";", "{", "}", "[", "]", "|", "\\", "<", ">", "+", "*", "=")):
        return True
    # 2. Purely numeric (e.g. "0", "007", "123", "2024", "404")
    # Strip common punctuation/symbols first to check if it's numeric
    numeric_check = text.replace(".", "").replace(",", "").replace("-", "").replace("%", "").strip()
    if numeric_check.isdigit():
        return True
    # 3. Starts with '0' (zero)
    if text.startswith("0"):
        return True
    # 4. Single-character punctuation or symbols
    if len(text) == 1 and not text.isalnum():
        return True
    return False


def source_matches(source: str, names: Set[str], prefixes: Set[str]) -> bool:
    return source in names or any(source.startswith(prefix) for prefix in prefixes)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_file = os.path.join(base_dir, "configs", "pipeline_config.yaml")
    input_file = os.path.join(base_dir, "datasets", "processed", "recipe_observations_v0.jsonl")
    output_file = os.path.join(base_dir, "datasets", "processed", "recipe_canonical_v0.jsonl")
    report_file = os.path.join(base_dir, "datasets", "reports", "clean_metrics.json")
    reject_samples_file = os.path.join(base_dir, "datasets", "reports", "quality_reject_samples.jsonl")

    # Load configuration
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "cleaning": {
                "lowercase_all": True,
                "split_ratio": {"train": 0.80, "dev": 0.10, "test": 0.10}
            }
        }

    clean_cfg = config.get("cleaning", {})
    lowercase_all = clean_cfg.get("lowercase_all", True)
    filter_garbage = clean_cfg.get("filter_garbage", True)
    filter_quality = clean_cfg.get("filter_quality", True)
    quality_config = quality_config_from_dict(clean_cfg.get("quality", {}))
    untrusted_sources = set(clean_cfg.get("untrusted_sources", ["expitau"]))
    untrusted_source_prefixes = set(clean_cfg.get("untrusted_source_prefixes", []))
    require_trusted_vocab_for_untrusted = clean_cfg.get("require_trusted_vocab_for_untrusted", True)
    reject_sample_limit = int(clean_cfg.get("reject_sample_limit", 500))
    split_ratios = clean_cfg.get("split_ratio", {"train": 0.80, "dev": 0.10, "test": 0.10})

    if not os.path.exists(input_file):
        print(f"Error: Input observations file not found at {input_file}")
        return

    trusted_concepts: Set[str] = set()
    if filter_quality and require_trusted_vocab_for_untrusted:
        print("Pass 1: Building trusted concept vocabulary...")
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obs = json.loads(line)
                except Exception:
                    continue
                source = obs.get("source", "")
                if source_matches(source, untrusted_sources, untrusted_source_prefixes):
                    continue
                for field in ("input_a", "input_b", "output"):
                    concept = normalize_concept(obs.get(field, ""))
                    if concept:
                        trusted_concepts.add(concept)

    print("Pass 2: Aggregating observations at the pair level...")
    
    # Track emojis for mode resolution
    emoji_counts = defaultdict(lambda: defaultdict(int))
    
    # Map pair_key -> output_name -> count
    pair_outputs = defaultdict(lambda: defaultdict(int))
    # Map pair_key -> set of sources
    pair_sources = defaultdict(set)

    line_count = 0
    garbage_discarded_count = 0
    quality_reject_counts = Counter()
    quality_reject_samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            line_count += 1
            if line_count % 2000000 == 0:
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

            # Normalize to lowercase
            key_a = input_a.lower()
            key_b = input_b.lower()
            key_out = output.lower()

            # Skip empty records
            if not key_a or not key_b or not key_out:
                continue

            if filter_garbage:
                if is_garbage_concept(key_a) or is_garbage_concept(key_b) or is_garbage_concept(key_out):
                    garbage_discarded_count += 1
                    continue

            if filter_quality:
                reject_reason = recipe_quality_reason(key_a, key_b, key_out, quality_config)
                if (
                    reject_reason == "ok"
                    and require_trusted_vocab_for_untrusted
                    and source_matches(source, untrusted_sources, untrusted_source_prefixes)
                    and (
                        key_a not in trusted_concepts
                        or key_b not in trusted_concepts
                        or key_out not in trusted_concepts
                    )
                ):
                    reject_reason = "untrusted_source_vocab"
                if reject_reason != "ok":
                    quality_reject_counts[reject_reason] += 1
                    if len(quality_reject_samples) < reject_sample_limit:
                        quality_reject_samples.append({
                            "input_a": key_a,
                            "input_b": key_b,
                            "output": key_out,
                            "source": source,
                            "reason": reject_reason
                        })
                    continue

            pair_key = (key_a, key_b) if key_a <= key_b else (key_b, key_a)

            # Record emoji counts
            if emoji_a and emoji_a != "⚪":
                emoji_counts[key_a][emoji_a] += 1
            if emoji_b and emoji_b != "⚪":
                emoji_counts[key_b][emoji_b] += 1
            if emoji_output and emoji_output != "⚪":
                emoji_counts[key_out][emoji_output] += 1

            # Accumulate
            pair_outputs[pair_key][key_out] += 1
            if source:
                pair_sources[pair_key].add(source)

    print("Resolving emoji modes...")
    resolved_emoji = {}
    for norm_name, emojis in emoji_counts.items():
        if emojis:
            resolved_emoji[norm_name] = max(emojis.items(), key=lambda x: x[1])[0]
        else:
            resolved_emoji[norm_name] = "⚪"

    print("Writing canonical pair-level recipes...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    metrics = {
        "num_canonical_pairs": 0,
        "num_conflicting_pairs": 0,
        "num_keep": 0,
        "num_review_identity": 0,
        "num_garbage_discarded": garbage_discarded_count,
        "num_quality_discarded": sum(quality_reject_counts.values()),
        "quality_reject_counts": dict(sorted(quality_reject_counts.items())),
        "split_counts": {"train": 0, "dev": 0, "test": 0}
    }

    # Sort keys of pair_outputs to guarantee deterministic ordering
    sorted_pairs = sorted(pair_outputs.keys())

    with open(output_file, "w", encoding="utf-8") as fout:
        for pair_key in sorted_pairs:
            key_a, key_b = pair_key
            
            known_outputs = sorted(list(pair_outputs[pair_key].keys()))
            known_emojis = [resolved_emoji.get(out, "⚪") for out in known_outputs]
            observation_counts = [pair_outputs[pair_key][out] for out in known_outputs]
            
            # Select canonical output based on highest observation count (mode)
            canonical_output = max(pair_outputs[pair_key].items(), key=lambda x: x[1])[0]
            canonical_emoji = resolved_emoji.get(canonical_output, "⚪")
            
            pair_num_outputs = len(known_outputs)
            is_conflicting = pair_num_outputs > 1

            if is_conflicting:
                metrics["num_conflicting_pairs"] += 1

            sources = sorted(list(pair_sources[pair_key]))
            split = assign_split(pair_key, split_ratios)

            # Determine quality status
            if canonical_output == key_a or canonical_output == key_b:
                status = "review_identity"
                metrics["num_review_identity"] += 1
            else:
                status = "keep"
                metrics["num_keep"] += 1

            metrics["num_canonical_pairs"] += 1
            metrics["split_counts"][split] += 1

            pair_id = stable_id([key_a, key_b])

            canonical_recipe = {
                "pair_id": pair_id,
                "pair_key": f"{key_a}+{key_b}",
                "input_a": key_a,
                "input_b": key_b,
                "emoji_a": resolved_emoji.get(key_a, "⚪"),
                "emoji_b": resolved_emoji.get(key_b, "⚪"),
                "known_outputs": known_outputs,
                "known_emojis": known_emojis,
                "canonical_output": canonical_output,
                "canonical_emoji": canonical_emoji,
                "observation_counts": observation_counts,
                "sources": sources,
                "source_count": len(sources),
                "pair_num_outputs": pair_num_outputs,
                "is_conflicting_pair": is_conflicting,
                "split": split,
                "status": status
            }

            fout.write(json.dumps(canonical_recipe, ensure_ascii=False) + "\n")

    # Write metrics report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as frep:
        frep.write(json.dumps(metrics, indent=2) + "\n")

    with open(reject_samples_file, "w", encoding="utf-8") as frej:
        for sample in quality_reject_samples:
            frej.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\nData canonicalization and splits assigned successfully!")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
