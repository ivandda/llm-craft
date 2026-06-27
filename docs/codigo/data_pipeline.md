# Data Cleaning, Splitting & Evaluation Pipeline

This document explains the deduplication, casing resolution, conflict flagging, stable hashing splits, and the SFT/evaluation export stages of the dataset pipeline.

---

## Pipeline Overview

The data pipeline processes raw normalized observations to produce deduplicated, split-versioned canonical recipes, minimal recipe splits for Supervised Fine-Tuning (SFT), and structured pairs for baseline evaluation. Prompt text is not stored in processed datasets; training and evaluation scripts render prompts at runtime.

```
[recipe_observations_v0.jsonl] (Bronze)
              ↓
      src/data/clean.py
              ↓
  [recipe_canonical_v0.jsonl] (Silver)
       ↙                    ↘
src/data/export_sft.py    src/data/export_eval.py
     ↓                            ↓
[recipes_{split}.jsonl]     [eval_{split}_{size}.jsonl]
```

Teacher enrichment is a later manual stage. It is intentionally not executed from
`run_pipeline.py` to avoid accidental API costs.

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

### 6. Garbage/Noise Concepts Filtering
To prevent vocabulary pollution and keep student training clean, the pipeline checks for noisy/corrupt concepts if `filter_garbage: true` is configured:
* **Discard Criteria**: A recipe is dropped if any concept (`input_a`, `input_b`, or `output`) meets any of the following:
  1. **Hashtags/Prefixes**: Contains `#`, backticks, or other code characters (e.g., `#bloodmoon`).
  2. **Purely Numeric**: Represents only digit characters (e.g. `0`, `007`, `123`).
  3. **Zero-Prefixed Noise**: Starts with `"0"` (e.g., `00b5`, `0bs`, `0ct`, `0faz`).
  4. **Suspicious Punctuation**: Contains characters like `;`, `{`, `}`, `[`, `]`, `|`, `\`, `<`, `>`, `+`, `*`.
  5. **Single-Character Punctuation**: Non-alphanumeric single-character strings (e.g. `;`, `.`).
* **Metrics**: The pipeline logs the count of dropped recipes under `"num_garbage_discarded"`.

### 7. Commonsense Quality Filtering
The pipeline now applies a stricter recipe-quality gate before canonical aggregation. The goal is to train and evaluate on clean commonsense combinations, not meme-like or malformed Infinite Craft artifacts.

The filter rejects a recipe if any input or output is:
* too long for a simple concept (`max_words` and `max_chars` in config),
* too short to be useful as a standalone concept,
* sentence-like or title-like,
* a formula/code fragment,
* numeric or digit-heavy,
* an identity copy,
* a placeholder/parse-artifact token such as `undefined`,
* a repeated-token phrase,
* a suspicious generated suffix such as `-nado`, or
* a vowelless long token that looks like an arbitrary acronym.

The filter also treats Expitau as an untrusted source by default. Expitau recipes are admitted only when their input and output concepts are already present in trusted sources, then pass the same general string-quality checks. This is intentionally high-precision: without a model judge or external knowledge base, regex alone cannot reliably distinguish a real named entity from a plausible-looking but bad compound.

For human review, cleaning writes:
* `datasets/reports/quality_reject_samples.jsonl`: capped rejected examples with one short reason.

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

We export recipe split files and dedicated evaluation files into `datasets/processed/`.

### 1. Minimal Recipe Exporter (`export_sft.py`)
Generates prompt-free recipe files:
* **`recipes_train.jsonl`**: clean training pairs.
* **`recipes_dev.jsonl`**: clean development pairs.
* **`recipes_test.jsonl`**: clean held-out test pairs.

Each row represents `A + B = {C, D, ...}`:

```json
{
  "input_a": "fire",
  "input_b": "water",
  "outputs": ["steam", "mist"]
}
```

`src/sft/train.py` expands `outputs` at runtime into supervised targets and renders the prompt from `--prompt-template`.

### 2. Structured Evaluation Exporter (`export_eval.py`)
Generates structured pair-level evaluation sets. It uses a **reservoir sampling** algorithm to scan the canonical dataset with low memory usage:
* **`eval_dev_all.jsonl`**: all clean dev pairs with the current config.
* **`eval_test_all.jsonl`**: all clean test pairs with the current config.

Set any evaluation size to `"all"` to export the complete matching split instead of a sample:

```yaml
evaluation_export:
  sizes:
    dev_keep: all
    test_keep: all
```

The resulting files use `all` in the name, for example `eval_dev_all.jsonl` and `eval_test_all.jsonl`. Other strings are rejected as config errors so typos such as `alll` do not silently create partial or empty eval sets.
Numeric sizes are also supported and use compact names such as `eval_dev_1k.jsonl`.

*Evaluation Record Schema:*
```json
{
  "input_a": "fire",
  "input_b": "water",
  "known_outputs": ["mist", "steam"]
}
```
*During scoring, a student prediction is evaluated against the entire list of `known_outputs` to reward legitimate alternative outputs.*

### 3. Teacher Multi-Output Enrichment (`enrich_multi_output.py`)
Generates grouped derived datasets under `datasets/enriched/` from the minimal recipe files.
This first enrichment stage does **not** include rationales. It preserves existing observed
outputs and calls the teacher only when a recipe has fewer than five valid outputs.

The recommended current path is `src/data/enrich_teacher.py`, which does the teacher review,
optional alternatives, and rationales in one structured-output call. The older two-step
scripts are still useful as smaller isolated stages, but the one-call path is simpler for
dataset construction and cost measurement.

### Recommended Structured Teacher Enrichment (`enrich_teacher.py`)
Generates `datasets/enriched/dataset_03_teacher_structured_enriched/` from the minimal
recipe split files. It uses Gemini 2.5 Flash-Lite by default and asks the teacher to:

* keep only observed outputs that make sense for the input pair,
* add new alternatives only when they are genuinely plausible,
* avoid forcing a fixed number of answers,
* write one short rationale for each accepted output,
* order accepted candidates from strongest to weakest recipe.

Records with fewer than `target_num_outputs` candidates are marked `partial_enrichment`.
This is expected and valid. Whole-recipe rejections are written to `rejected.jsonl`, not the
enriched split file. The script writes a manifest with token usage and estimated realtime
Flash-Lite cost. Candidate order is stored as a programmatic `rank` field. The teacher does
not emit numeric scores.

```bash
# Development smoke test
uv run python -m src.data.enrich_teacher --splits train --limit 10 --no-resume

# Cost/quality sample before a larger run
uv run python -m src.data.enrich_teacher --splits train --limit 100 --no-resume
```

Output folders:
* **`dataset_00_recipes_baseline_multi_output/`**: organized copy of `recipes_train/dev/test.jsonl`.
* **`dataset_01_teacher_enriched_multi_output_no_rationale/`**: grouped candidate outputs with
  `source: "observed"` or `source: "teacher"`.

Example enriched record:

```json
{
  "input_a": "fire",
  "input_b": "water",
  "candidate_outputs": [
    {"output": "steam", "source": "observed"},
    {"output": "mist", "source": "observed"},
    {"output": "vapor", "source": "teacher"},
    {"output": "hot spring", "source": "teacher"},
    {"output": "sauna", "source": "teacher"}
  ],
  "quality_status": "complete",
  "metadata": {
    "source_dataset": "recipes",
    "source_split": "train",
    "teacher_provider": "google_vertex_ai",
    "teacher_model": "gemini-2.5-flash",
    "enrichment_version": "teacher_multi_output_no_rationale_v1",
    "target_num_outputs": 5,
    "has_rationales": false
  }
}
```

Run `enrich_multi_output.py` from the repository root. This step calls the teacher model for
recipes that need additional candidate outputs.

```bash
# Minimal real smoke runs by split
uv run python -m src.data.enrich_multi_output --splits train --limit 3
uv run python -m src.data.enrich_multi_output --splits dev --limit 3
uv run python -m src.data.enrich_multi_output --splits test --limit 3

# Regenerate a small split sample from scratch
uv run python -m src.data.enrich_multi_output --splits train --limit 3 --no-resume

# Run one complete split
uv run python -m src.data.enrich_multi_output --splits train
uv run python -m src.data.enrich_multi_output --splits dev
uv run python -m src.data.enrich_multi_output --splits test

# Run all splits, resuming already generated records
uv run python -m src.data.enrich_multi_output

# Run all splits from scratch
uv run python -m src.data.enrich_multi_output --no-resume
```

`--resume` is enabled by default. With resume enabled, the script skips input pairs already
present in the output split file and does not call the LLM for those pairs. Use `--no-resume`
only when you intentionally want to overwrite and regenerate the selected output files.

### 4. Teacher Rationale Enrichment (`enrich_rationales.py`)
Adds one concise rationale to each existing `candidate_output` in
`dataset_01_teacher_enriched_multi_output_no_rationale/`. This stage does not generate new
outputs: it validates that the teacher returns the same output strings in the same order and
records failures when the response is invalid or incomplete.

Output folder:
* **`dataset_02_teacher_enriched_multi_output_with_rationale/`**: same candidates as dataset 01,
  with a `rationale` field added to each candidate when generation succeeds.

Example rationale-enriched record:

```json
{
  "input_a": "fire",
  "input_b": "water",
  "candidate_outputs": [
    {
      "output": "steam",
      "source": "observed",
      "rationale": "Fire heats water until it turns into steam."
    },
    {
      "output": "sauna",
      "source": "teacher",
      "rationale": "Heat and water combine into the hot, steamy setting of a sauna."
    }
  ],
  "quality_status": "complete",
  "metadata": {
    "source_dataset": "dataset_01_teacher_enriched_multi_output_no_rationale",
    "source_split": "train",
    "teacher_provider": "google_vertex_ai",
    "teacher_model": "gemini-2.5-flash",
    "enrichment_version": "teacher_multi_output_with_rationale_v1",
    "target_num_outputs": 5,
    "has_rationales": true,
    "rationale_language": "en"
  }
}
```

Run `enrich_rationales.py` after `dataset_01_teacher_enriched_multi_output_no_rationale/`
exists. This step calls the teacher model to explain existing candidates; it does not create
new outputs.

```bash
# Minimal real smoke runs by split
uv run python -m src.data.enrich_rationales --splits train --limit 3
uv run python -m src.data.enrich_rationales --splits dev --limit 3
uv run python -m src.data.enrich_rationales --splits test --limit 3

# Regenerate a small split sample from scratch
uv run python -m src.data.enrich_rationales --splits dev --limit 3 --no-resume

# Run one complete split
uv run python -m src.data.enrich_rationales --splits train
uv run python -m src.data.enrich_rationales --splits dev
uv run python -m src.data.enrich_rationales --splits test

# Run all splits, resuming already completed rationale records
uv run python -m src.data.enrich_rationales

# Run all splits from scratch
uv run python -m src.data.enrich_rationales --no-resume
```

For rationale enrichment, resume skips a record only when the output already contains all
rationales for that input pair. If a record is missing any rationale, the script will call the
LLM again for that record.

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

# Step 3: Export minimal recipe split datasets
uv run python -m src.data.export_sft

# Step 4: Export structured evaluation datasets
uv run python -m src.data.export_eval
```
