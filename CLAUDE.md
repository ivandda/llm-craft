# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`llm-craft` distills *compositional creativity* into small LLMs, inspired by *Infinite Craft*: a recipe is a pair of input concepts (`input_a`, `input_b`) that combine into one or more output concepts (e.g. `fire + water -> steam`). The repo covers the full loop: build a clean recipe dataset from raw game dumps, enrich it with a teacher LLM (Gemini), fine-tune a student with SFT/QLoRA, and evaluate it with a compositional creativity score. `AGENTS.md` holds generic behavioral guidelines; documentation lives in `docs/` and most prose (README, docs) is in Spanish.

## Environment & tooling

- Python is managed with **uv** (`requires-python >=3.12`). Always run code via `uv run ...`. Install with `uv sync`.
- Dependency groups (see `pyproject.toml`): `dev`+`agents` install by default. The `vertex` group is opt-in (`uv sync --group vertex`) and is only needed to submit Vertex jobs or run Vertex-judged evaluation. Keep local training lean by not depending on `vertex`/`agents` from the training/eval core.
- Vertex/GCP config comes from env (`.env`, see `.env.example`): `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT=nlp2026-498021`. GCP project is `nlp2026-498021`, bucket `llm-craft-bucket`, Artifact Registry repo `llm-craft-registry`.

## Common commands

Tests (the plugin-autoload disable avoids loading unrelated global pytest plugins):
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests -q          # all
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/sft -q      # one dir
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/sft/test_losses.py::test_name -q  # single test
RUN_SFT_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/sft/test_smoke_train.py -q  # opt-in, downloads a model
```

Data pipeline (config-driven by `configs/pipeline_config.yaml`):
```bash
uv run python -m src.data.run_pipeline   # runs normalize -> clean -> export_sft -> export_eval
uv run python -m src.data.normalize      # or run any single stage module
```

SFT training (config-driven; CLI flags override YAML fields one-to-one):
```bash
uv run python -m src.sft.train --config configs/sft/default.yaml --run_name my_run
```

Creativity eval on a trained adapter (needs the `vertex` group when using the LLM judge):
```bash
uv run --group vertex python -m src.eval.run_sft_eval --run_dir runs/sft/<run_id> --eval_file datasets/final-10k/test.jsonl
```

Frontend (`apps/web`, Next.js):
```bash
cd apps/web && npm ci && npm run dev
npm run typecheck && npm run test && npm run build   # validate
```

## Architecture

The Python code under `src/` is four independent subsystems. Each is a set of `python -m src.<pkg>.<module>` entry points; they communicate only through JSONL files on disk under `datasets/` and `runs/`.

### `src/data/` — dataset pipeline (medallion-style layers)
Raw game dumps in `datasets/raw/{ericlewis,elementia,expitau,redfast00}` → normalized "bronze" observations → cleaned "silver" canonical recipes with deterministic train/dev/test splits → "gold" minimal SFT/eval exports. Key properties:
- **Splits are assigned by hashing the input pair**, so the same `(input_a, input_b)` never leaks across splits.
- Prompts are **not** stored in `datasets/processed`; they are injected at runtime by the training/eval code.
- Which raw sources and quality filters are active is controlled entirely by `configs/pipeline_config.yaml` (`raw_datasets` toggles, `quality` rules, `split_ratio`, `exclude_identities`, eval sizes).
- Teacher enrichment (`enrich_teacher.py`, `enrich_rationales.py`) uses **Gemini 2.5 Flash** and is a **manual** step, not part of `run_pipeline`. It supports realtime and Vertex **batch** modes (`--mode batch-export|batch-submit|batch-import`, ~50% cheaper). It preserves observed outputs and only fills plausible alternatives; partial records are valid.

### `src/sft/` — SFT / (Q)LoRA training
Trains causal LMs on JSONL recipes with **multiple weighted candidate outputs per recipe**. Central design point is a **single unified loss family** parameterized by two axes (`losses.py`):
- `candidate_weighting`: `uniform` | `dataset` (per-candidate weights from the data)
- `candidate_aggregation`: `expected_logprob` (soft-CE) | `logsumexp_prob` (concept-set)
- `loss_type` (`ce`, `soft_ce`, `concept_set`, `concept_set_uniform`) is a **legacy alias** that expands to a `(weighting, aggregation)` pair. If you set either axis explicitly you must set both; you can't mix an alias with a partial override.
- Loss is computed **only over the final-concept tokens**, never the prompt. The `collator.py` builds prompts + a `concept_mask` via token offsets and keeps all candidates of one recipe in the same batch — `per_device_*_batch_size` counts **recipes, not candidates**.
- Optional auxiliary rationale term (`rationale_loss_weight`, `rationale_position=output_before_rationale|output_after_rationale`) trains on candidate explanations when present.
- `config.py` (`SFTConfig` dataclass) loads YAML + CLI overrides. `train.py` is the entry point; it writes a self-contained `runs/sft/<timestamp>_<run_name>/` with `config.yaml`, `command.txt`, `git_info.json`, `metrics.jsonl`, checkpoints, `best_adapter/`, `final_adapter/`, and loss plots. Supports `--resume_from_checkpoint`.
- Default config (`configs/sft/default.yaml`) is a CPU-viable **smoke test** on `sshleifer/tiny-gpt2`. Real runs use Qwen (`qwen*_example.yaml`, e.g. `Qwen/Qwen3-4B-Instruct-2507`) and need CUDA. `prompt_format: qwen_chat` switches to the tokenizer chat template.

### `src/eval/` — creativity evaluation
`run_sft_eval.py` loads a `run_dir`, generates `K` samples per input, and computes an operational **Compositional Creativity Score**: `C(x) = α·mean(q^λ·n) + (1−α)·d`, where `q`=plausibility, `n`=novelty, `d`=diversity. Embeddings are pluggable (`--embedding_backend glove|word2vec|sentence_embeddings`); novelty can use a **Vertex LLM judge** (Anthropic Claude on Vertex, `vertex_judge.py`) with one call per recipe.

### `src/agent/` — playable agent runner (uses the `agents` dep group: langchain + google-genai + deepagents).

### `apps/web/` — Next.js frontend (mock-only, typed contracts for connecting models later; seeded creds `admin/admin`).

## Vertex AI training

The same `src/sft/train.py` runs unmodified as a Vertex **Custom Training Job**: Vertex mounts GCS at `/gcs/<bucket>/`, so `datasets/` reads and `runs/` writes go through the mount. The `Dockerfile` bakes `configs/sft/*.yaml` into the image; `vertex_submit.py` rewrites data/output paths to the mount and can run other modules via `--module` (e.g. `src.eval.run_sft_eval`). Flow: `gsutil cp` data → `gcloud builds submit --config cloudbuild.yaml` → `uv run --group vertex python -m src.sft.vertex_submit --run-name X --config ...` (or `./scripts/run_vertex_qwen3_thinking.sh`). Full details in `docs/codigo/vertex_training.md`.

## Conventions

- All inter-stage data is JSONL. The canonical training row is `{"input_a", "input_b", "candidate_outputs":[{"output","source","rank","weight?","rationale?"}]}`; legacy `outputs`/`output` rows are adapted internally.
- Behavior is driven by YAML config + explicit CLI overrides, not hardcoded constants — prefer extending the config over branching in code.
- Per `AGENTS.md`: keep changes surgical, don't refactor unrelated code, prefer the simplest solution, and update `docs/` + `README.md` when you change behavior.
