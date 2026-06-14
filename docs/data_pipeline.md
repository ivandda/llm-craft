# Data Cleaning & Splitting Pipeline

This document explains the deduplication, casing resolution, conflict flagging, deterministic splitting, and conversational SFT export stages of the dataset pipeline.

---

## Pipeline Overview

The data pipeline processes raw normalized observations to produce deduplicated, split-versioned canonical recipes, and outputs conversational formats suitable for Supervised Fine-Tuning (SFT).

```
[recipe_observations_v0.jsonl] (Bronze)
              ↓
      src/data/clean.py
              ↓
  [recipe_canonical_v0.jsonl] (Silver)
              ↓
    src/data/export_sft.py
              ↓
[sft_train/dev/test.jsonl] (Gold)
```

---

## Cleaning & Canonicalization Design

The cleaning stage (`src/data/clean.py`) uses a memory-efficient **two-pass approach** with standard libraries to process the 6.7M observations:

### 1. Casing Mode Resolution
Raw inputs contain inconsistent casings (e.g. `steam`, `Steam`, `STEAM`).
* **Logic**: In Pass 1, we count occurrences of each capitalization form for every unique concept. In Pass 2, we resolve each concept to its **most frequent casing (mode)**. This ensures clean, uniform capitalization (e.g., `Fire`, `Steam`, `Water`) in SFT prompts.

### 2. Emoji Mode Selection
Similarly, if different sources provide different emojis for the same concept, we resolve the concept's emoji to the most frequent non-generic emoji (ignoring fallback `⚪` where possible).

### 3. Duplicate Aggregation
Identical recipes `(input_a, input_b, output)` are grouped into a single canonical recipe. We compute the `observation_count` (total occurrences) and track the list of `sources` (e.g. `["ericlewis_train", "expitau"]`).

### 4. Conflict Flagging
* **Logic**: We group combinations by `(input_a, input_b)` to find pairs that produce multiple outputs (e.g., `Fire + Water` producing both `Steam` and `Mist`).
* **Flagging**: We set `is_conflicting_pair = true` and record the count of unique outputs as `pair_num_outputs`.
* **Conceptual & Research Relevance**:
  * **Creative Alternatives (Not Data Noise)**: In typical classification tasks, a conflicting pair (same inputs, different labels) is treated as noise. In compositional creativity tasks like *Infinite Craft*, conflicts represent legitimate **alternative creative outputs** (e.g., `Fire + Water` can yield `Steam`, `Mist`, `Hot Spring`, or `Sauna` depending on the game or dataset).
  * **Preference Tuning (DPO / PPO)**: These conflicting pairs serve as the baseline candidate pools for preference learning. A teacher model can later evaluate conflicting pairs (e.g. comparing the plausibility or novelty of `Steam` vs. `Mist`) to generate `chosen` and `rejected` responses for DPO dataset construction.
  * **SFT Target Decisions**: The presence of `is_conflicting_pair` allows downstream trainers to decide whether to train the student on all outputs (for diversity) or filter to the most frequent output (to avoid gradient conflicts during SFT).

### 5. Quality Status Labeling
Each recipe is tagged with a `status`:
* `"keep"`: Standard deduplicated recipe.
* `"keep_conflicting"`: Legitimate creative alternative output.
* `"review_identity"`: Output equals one of the inputs (e.g., `Water + Fire = Water`), flagged for manual review or future pruning.
* `"drop_empty_output"`: Missing output.

---

## Split Assignment & Input Leakage Prevention

To ensure a robust evaluation of **compositional creativity**, we perform a deterministic split using hashing:

* **Hashing Mechanism**: We compute `sha256(input_a_norm + "+" + input_b_norm)` and assign the split based on the modulo of the resulting integer:
  * **Train (80%)**: Modulo `< 8000`
  * **Dev (10%)**: Modulo `8000–8999`
  * **Test (10%)**: Modulo `9000–9999`
* **Why this is critical**: Hashing by input **pair** (and not individual recipe) ensures that all outputs of `Fire + Water` land in the *same split*. This prevents the model from learning `Fire + Water = Steam` in training and being tested on the leaked input pair in `Fire + Water = Mist`.

---

## Conversational SFT Export Format

The export script (`src/data/export_sft.py`) writes conversational training items for splits to `datasets/processed/sft_{split}.jsonl` using a standard `messages` structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Combine the concepts: Fire + Water. Return only the resulting concept."
    },
    {
      "role": "assistant",
      "content": "Steam"
    }
  ],
  "metadata": {
    "pair_key": "fire+water",
    "recipe_key": "fire+water=>steam",
    "source_count": 2,
    "is_conflicting_pair": true,
    "split": "train"
  }
}
```

---

## Validation and Split Metrics

Executing the pipeline outputs the following metrics:

* **Canonical Recipes**: `6,547,063`
* **Unique Input Pairs**: `6,541,348`
* **Conflicting Input Pairs**: `5,007` (yielding `10,486` alternative recipes)
* **Identity Review Recipes**: `511,596`
* **Split Counts**:
  * **Train**: `5,237,309` (80.0%)
  * **Dev**: `654,881` (10.0%)
  * **Test**: `654,873` (10.0%)

---

## Future Integration with LLM Teacher Augmentation

In the next phase of the project, teacher models (e.g. Gemini) will generate **rationales**, **negatives**, and **preference scores** to enable M3 (SFT + Rationale) and M4 (DPO) training. 

* **Joinable Augmentation Table**: These teacher-enriched records will be stored in `datasets/processed/teacher_enriched_v0.jsonl` and joined with the canonical split datasets using `recipe_key` (or `pair_key`). This prevents bloating the base canonical table and avoids training leakage.
