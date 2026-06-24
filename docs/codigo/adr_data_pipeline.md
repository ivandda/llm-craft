# Architecture Decision Record (ADR): Data Pipeline Design

This document details the architectural decisions made during the design and implementation of the `llm-craft` data normalization, cleaning, and export pipeline.

---

## Status
**Accepted** (June 2026)

---

## Decisions Summary

### 1. Lowercase Normalization
* **Context**: Raw datasets contain inconsistent capitalization (e.g., `steam`, `Steam`, `STEAM`), which fragments the vocabulary for small student LLMs and complicates training.
* **Decision**: Normalize all concept strings, inputs, and outputs to lowercase in the canonical database and training/evaluation sets.
* **Consequences**: Unifies vocabularies and simplifies semantic composition learning. Casing displays must now be handled as a post-processing step (e.g., `.title()`).

---

### 2. Pair-Level Database Aggregation
* **Context**: Multiple sources might link the same input pair to different creative outputs. Storing these as separate rows duplicates pair metadata and complicates indexing.
* **Decision**: Collapse the canonical recipe database (`recipe_canonical_v0.jsonl`) to exactly one row per unique input pair. We store all valid outputs in a `known_outputs` array and select the most frequent output as `canonical_output`.
* **Consequences**: Shrinks database size, natively supports alternative outputs, and facilitates multi-answer validation.

---

### 3. Cryptographic Hash IDs (`pair_id`, `recipe_id`)
* **Context**: String delimiters (e.g., `fire+water` or `fire+water=>steam`) collide when concept names contain special characters (like `C++` or `Java => Kotlin`).
* **Decision**: Generate unique IDs using `sha256` hashing on sorted, JSON-serialized concept lists.
* **Consequences**: Guarantee collision-free database indices. Human-readable string keys are kept only as debug metadata.

---

### 4. Deterministic Hash-Based Split Assignment
* **Context**: Random splits cause input leakage. If `fire+water=>steam` is in train, and `fire+water=>mist` is in test, the model is tested on an input combination it has already seen during training, ruining Out-Of-Distribution (OOD) evaluation.
* **Decision**: Assign splits deterministically using `sha256(input_a_norm + "+" + input_b_norm) % 10000`.
* **Consequences**: Ensures zero input-pair leakage between splits. Splits are identical across different machines and runs without requiring Git-tracked split files.

---

### 5. Exclusion of Identity Copies from SFT Clean Baseline
* **Context**: Recipes where the output is identical to one of the inputs (e.g., `Jacket + Worse = Jacket`) teach the model to cheat by lazy-copying the input when confused.
* **Decision**: Mark these recipes as `review_identity` and exclude them from the strict recipe training baseline.
* **Consequences**: Forces the student model to learn actual semantic combination, though it reduces the size of the training pool by ~7.8%.

---

### 6. Streaming Reservoir Sampling for Evaluation Sets
* **Context**: Loading 6.5M recipes into memory to draw a random sample of 1,000 items exceeded 10 GB of RAM, causing system freezes (paging).
* **Decision**: Implement a single-pass streaming Reservoir Sampling algorithm.
* **Consequences**: Reduces RAM usage to less than 10 MB, executing the full evaluation set export in under 20 seconds.

---

### 7. Centralized Configuration and In-Process Pipeline Orchestration
* **Context**: Hardcoding splits, filter rules, prompt templates, and execution order across multiple scripts hurts reproducibility.
* **Decision**: Create a single `configs/pipeline_config.yaml` and a master orchestrator `src/data/run_pipeline.py` that imports and runs all steps in-process.
* **Consequences**: Promotes reproducibility (clone $\rightarrow$ config $\rightarrow$ run). In-process function execution is faster than spawning python subprocesses and shares common schema imports.

---

### 8. Garbage/Noise Concepts Filtering
* **Context**: Raw logs contain automatic crafter bot noise (e.g. hashtags `#bloodmoon`, numeric elements `0`, `007`, `0002`, `00b5`, and syntax garbage like `0;`). Fine-tuning on these pollutes the model's vocabulary and composition skills.
* **Decision**: Drop any recipe where either `input_a`, `input_b`, or the output matches garbage patterns: starting with `0`, purely numeric, containing invalid code characters (`#`, `;`, `{`, `}`, `[`, `]`, `|`, `\`, `+`, `*`), or single-character punctuation.
* **Consequences**: Safely prunes noise from all downstream training and evaluation splits, ensuring a clean student model. Discarded 48k+ recipes.

---

### 9. Configurable & Dynamic Evaluation Sizes
* **Context**: Hardcoding evaluation sizes (`1k` and `500` samples) directly in filenames makes config modifications misleading.
* **Decision**: Allow the user to configure exact target sizes under `evaluation_export: sizes` in the config YAML, and update `export_eval.py` to dynamically name files using a size formatter (e.g., `eval_dev_1k.jsonl` or `eval_test_2k.jsonl`).
* **Consequences**: Synchronizes config sizes with output filenames automatically, preventing mismatches and improving pipeline flexibility.

---

### 10. Commonsense Quality Gate for SFT/Evaluation
* **Context**: Expitau is large but contains many malformed or game-meme concepts, including formulas, random acronyms, long sentence-like titles, and plausible-looking but bad compounds. These examples are bad targets for a clean commonsense SFT baseline.
* **Decision**: Add `src/data/quality.py` and make cleaning reject recipes whose inputs or outputs fail configurable concept-shape checks. Treat Expitau as untrusted by default: an Expitau recipe must use concepts already present in trusted non-Expitau sources before it can enter SFT/evaluation.
* **Consequences**: Expitau can be enabled as a high-precision augmentation source without pretending that regex can solve semantic plausibility. The pipeline writes minimal prompt-free recipe artifacts (`recipes_{split}.jsonl`, `eval_{split}_{size}.jsonl`, and `quality_reject_samples.jsonl`) so humans can quickly inspect accepted and rejected examples.

---

### 11. Prompt-Free Processed Datasets
* **Context**: Storing full chat prompts in `datasets/processed` created many near-duplicate files and mixed data facts with runtime presentation choices.
* **Decision**: Store only recipe facts in processed datasets: `input_a`, `input_b`, and `outputs`/`known_outputs`. Training and evaluation scripts render prompts at runtime from configurable templates.
* **Consequences**: The processed directory is easier to audit (`A+B={C,D}`), prompt experiments do not require regenerating datasets, and runtime prompt injection remains isolated to model-facing code.
