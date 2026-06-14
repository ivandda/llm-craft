# Data Normalization Process

This document describes how the raw datasets were normalized and consolidated into a single unified recipe dataset located at `datasets/processed/recipe_observations_v0.jsonl`.

## Datasets and Raw Formats

We consolidated four distinct datasets, each with unique schemas and formatting styles:

1. **Ericlewis (`datasets/raw/eirclewis/data/`)**:
   - **Format**: JSON Lines containing training/validation/test sets designed for LLM instruction tuning.
   - **Key extraction**: Extracted user message contents (separated by ` + `) as inputs and assistant responses (containing JSON representation of results and emojis) as outputs.
2. **Elementia (`datasets/raw/elementia/`)**:
   - **Format**: A CSV file (`recipes.csv`) containing combinations.
   - **Key extraction**: Split row elements by `+` to extract the two inputs and mapped the second column directly to the output. Emojis were initially missing.
3. **Expitau (`datasets/raw/expitau/`)**:
   - **Format**: A large data package (`data.json`) where recipes are mapped by hashed keys to a lookup index.
   - **Key extraction**: Loaded the lookup index mapping hash keys to `[emoji, name, cost]` and translated the semicolon-separated recipe combinations into human-readable element names and emojis.
4. **Redfast00 (`datasets/raw/redfast00/`)**:
   - **Format**: Several JSON files representing different Alchemy games in the `JSONrecipes/` folder.
   - **Key extraction**: Loaded name mappings and lists of combinations containing lists of ingredient IDs and output IDs. Flattened recipes with multiple outputs into individual single-output observations.

---

## Normalization & Canonicalization Rules

To produce a single clean dataset suitable for downstream modeling, cleaning, and augmentation, the normalization pipeline implements three core rules:

### 1. Commutative Ingredient Sorting
Since combination recipes in Alchemy games are commutative (e.g. Combining `A + B` produces the same result as `B + A`), all input pairs are sorted alphabetically. 
* *Rule*: `input_a` is always alphabetically less than or equal to `input_b` case-insensitively.

### 2. Emoji Propagation & Backfilling
Emojis were sparse across the raw datasets (completely missing in `elementia` and `redfast00`).
* *Rule*: During parsing, the pipeline extracts every unique element-to-emoji mapping from source datasets (`ericlewis` and `expitau`). It compiles a global case-insensitive `emoji_map`. In the second pass, it backfills the emojis for all records where the ingredient or output name matches a known element.

### 3. Display Casing Preservation
While deduplication and conflict checks are performed case-insensitively, the raw display casing is preserved in the output records to maintain readable formatting.

---

## Output Schema

The final output is saved to `datasets/processed/recipe_observations_v0.jsonl` as JSON Lines. Each line matches the following schema:

```json
{
  "input_a": "Fire",
  "input_b": "Water",
  "output": "Steam",
  "emoji_a": "🔥",
  "emoji_b": "💧",
  "emoji_output": "💨",
  "source": "ericlewis_train"
}
```

---

## Validation Statistics

The dataset was validated using `src/data/validate.py` yielding the following statistics:

* **Total Observations**: `6,713,097`
* **Unique Combinations (Input A, Input B)**: `6,541,348`
* **Unique Recipes (Input A, Input B, Output)**: `6,547,063`
* **Unique Sources**: `12`
* **Parse Error Rate**: `0.0%`
* **Duplicate Recipe Count**: `166,034` (Identical recipes occurring across multiple sources or splits)
* **Conflicting Pair Count**: `5,007` (The same pair of inputs yielding different outputs, e.g. due to game variations)

---

## Running the Pipeline

To re-run the normalization pipeline or check validation stats, run the following commands:

```bash
# Normalize the datasets
uv run python -m src.data.normalize

# Validate the generated dataset
uv run python -m src.data.validate
```
