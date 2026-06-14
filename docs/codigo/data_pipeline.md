# Data Cleaning, Splitting & Evaluation Pipeline

This document explains the deduplication, casing resolution, conflict flagging, stable hashing splits, and the SFT/evaluation export stages of the dataset pipeline.

---

## Pipeline Overview

The data pipeline processes raw normalized observations to produce deduplicated, split-versioned canonical recipes, and outputs conversational formats suitable for Supervised Fine-Tuning (SFT) as well as structured pairs for baseline evaluation.

```
[recipe_observations_v0.jsonl] (Bronze)
              ↓
      src/data/clean.py
              ↓
  [recipe_canonical_v0.jsonl] (Silver)
       ↙                    ↘
src/data/export_sft.py    src/data/export_eval.py
     ↓                            ↓
[sft_{var}_{split}.jsonl]   [eval_{split}_{size}.jsonl]
```

---

## Cleaning & Canonicalization Design

The cleaning stage (`src/data/clean.py`) uses a memory-efficient **two-pass approach** with standard libraries to process the 6.7M observations:

### 1. Casing and Lowercase Normalization
To simplify student model learning, all elements, inputs, and outputs are normalized to **lowercase** throughout the cleaning and SFT/evaluation export stages. This prevents vocabulary fragmentation (e.g. treating `steam`, `Steam`, and `STEAM` as separate tokens) and allows the student model to focus entirely on compositional semantics. Raw display casing is preserved in the Bronze layers for provenance tracking but collapsed in the Silver/Gold layers.

### 2. Emoji Mode Selection
Similarly, if different sources provide different emojis for the same concept, we resolve the concept's emoji to the most frequent non-generic emoji (ignoring fallback `⚪` where possible).

### 3. Stable Hash IDs (`pair_id` and `recipe_id`)
To prevent string collision errors when concepts contain special characters (like `+` or `=>` e.g. `C++`), the pipeline generates unique cryptographic hash keys:
* `pair_id`: `sha256(json_array([input_a_norm, input_b_norm]))`
* `recipe_id`: `sha256(json_array([input_a_norm, input_b_norm, output_norm]))`

These IDs are used for all internal splitting and joining operations, while keeping raw display keys like `pair_key` (`fire+water`) for debugging and inspection.

### 4. Conflict Flagging
* **Logic**: We group combinations by `(input_a, input_b)` to find pairs that produce multiple outputs (e.g., `Fire + Water` producing both `Steam` and `Mist`).
* **Flagging**: We set `is_conflicting_pair = true` and record the count of unique outputs as `pair_num_outputs`.
* **Conceptual & Research Relevance**:
  * **Creative Alternatives (Not Data Noise)**: In typical classification tasks, a conflicting pair (same inputs, different labels) is treated as noise. In compositional creativity tasks like *Infinite Craft*, conflicts represent legitimate **alternative creative outputs** (e.g., `Fire + Water` can yield `Steam`, `Mist`, `Hot Spring`, or `Sauna` depending on the game or dataset).
  * **Preference Tuning (DPO / PPO)**: These conflicting pairs serve as the baseline candidate pools for preference learning. A teacher model can later evaluate conflicting pairs (e.g. comparing the plausibility or novelty of `Steam` vs. `Mist`) to generate `chosen` and `rejected` responses for DPO dataset construction.
  * **SFT Target Decisions**: The presence of `is_conflicting_pair` allows downstream trainers to decide whether to train the student on all outputs (for diversity) or filter to the most frequent output (to avoid gradient conflicts during SFT).

### 5. Quality Status Labeling
Each recipe is tagged with a status:
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

## Dataset Variants & Outputs

We export SFT variants and dedicated evaluation files into `datasets/processed/`.

### 1. Conversational SFT Exporter (`export_sft.py`)
Generates training files in conversational `messages` format:
* **`sft_clean`**: Strict, high-confidence recipes. Excludes conflicts and identity copies (`status == "keep" and is_conflicting_pair == false`).
  * `train`: `4,823,461` recipes
  * `dev`: `603,384` recipes
  * `test`: `603,103` recipes
* **`sft_all`**: All valid recipes (`status in ["keep", "keep_conflicting", "review_identity"]`).
  * `train`: `4,827,910` recipes
  * `dev`: `603,929` recipes
  * `test`: `603,628` recipes

*Prompt Format:*
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Given two concepts, combine them into one resulting concept.\n\nConcept A: Fire\nConcept B: Water\n\nReturn only the resulting concept."
    },
    {
      "role": "assistant",
      "content": "Steam"
    }
  ],
  "metadata": {
    "pair_id": "...",
    "recipe_id": "...",
    "pair_key": "fire+water",
    "recipe_key": "fire+water=>steam",
    "source_count": 2,
    "observation_count": 4,
    "pair_num_outputs": 1,
    "is_conflicting_pair": false,
    "status": "keep",
    "split": "train"
  }
}
```

### 2. Structured Evaluation Exporter (`export_eval.py`)
Generates structured pair-level evaluation sets. It uses a **reservoir sampling** algorithm to scan the 3.5 GB dataset in two streaming passes using **less than 10 MB of RAM**:
* **`eval_dev_1k.jsonl`**: 1,000 clean dev pairs.
* **`eval_test_1k.jsonl`**: 1,000 clean test pairs.
* **`eval_test_identity_500.jsonl`**: 500 test identity pairs (output equals input).
* **`eval_test_conflicting_500.jsonl`**: 480 test conflicting pairs (pairs with multiple known correct outputs; 480 represents the total number of conflicts in the test split).

*Evaluation Record Schema:*
```json
{
  "pair_id": "8a38a7c2921a8d0526be...",
  "pair_key": "fire+water",
  "input_a": "Fire",
  "input_b": "Water",
  "known_outputs": ["Mist", "Steam"],
  "canonical_output": "Steam",
  "status": "keep",
  "split": "test"
}
```
*During scoring, a student prediction is evaluated against the entire list of `known_outputs` to reward legitimate alternative outputs.*

---

## Running the Pipeline

To re-run the pipeline or export datasets, you can run the master orchestrator script which executes all steps sequentially:

```bash
uv run python -m src.data.run_pipeline
```

Alternatively, you can run individual steps:

```bash
# Step 1: Normalize observations
uv run python -m src.data.normalize

# Step 2: Clean, canonicalize and assign splits
uv run python -m src.data.clean

# Step 3: Export SFT datasets
uv run python -m src.data.export_sft

# Step 4: Export structured evaluation datasets
uv run python -m src.data.export_eval
```
