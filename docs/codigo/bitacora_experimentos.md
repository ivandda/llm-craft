# Bitácora de experimentos — SFT en Vertex AI

Registro de lo que se entrenó y evaluó hasta ahora en GCP, reconstruido a partir de
los `args` de los Custom Jobs, los `run_dir` en el bucket y las métricas guardadas.
El objetivo es que el estado del proyecto quede claro y no haya que reconstruirlo
de memoria. Última actualización: 2026-07-06.

> Nota: la mayoría de las conclusiones de "ranking" NO se pueden sacar todavía
> (ver [Qué falta y advertencias](#qué-falta-y-advertencias)). Esta bitácora
> documenta *qué se corrió*, no *qué loss ganó*.

## Infraestructura

- Proyecto GCP: `nlp2026-498021` (número `486944883203`).
- Bucket: `gs://llm-craft-bucket` (datos en `datasets/`, corridas en `runs/`, evals en `eval_outputs/`).
- Región: `us-central1`. Hardware: 1× **NVIDIA L4** (`g2-standard-8`).
- Modelo base: **`Qwen/Qwen3-4B-Thinking-2507`**.
- Imagen de training en Artifact Registry (`llm-craft-registry/llm-craft-sft`), lanzada con `src/sft/vertex_submit.py`.

## Configuración común de entrenamiento

Todas las corridas 4B parten de `configs/sft/qwen3_4b_thinking_10k_example.yaml` y
solo difieren en los ejes de la loss (pasados como override por CLI):

- QLoRA 4-bit (nf4), LoRA `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=auto`.
- 3 épocas → **30.495 steps**, `lr=2e-4`, `bf16`, gradient checkpointing.
- `prompt_format: qwen_chat`, `max_seq_length=512`, `seed=42`.
- `length_normalize_concept_logprob: true` (ver advertencias).
- Dataset: `datasets/final-10k` (`train.jsonl` + `dev.jsonl`, `max_dev_examples=1291`).

Aclaraciones que suelen generar confusión:

- **Base = versión Thinking**: las 4 usan `Qwen/Qwen3-4B-Thinking-2507` como
  checkpoint base, pero la supervisión es **solo sobre el concepto final** (sin
  `<think>` ni rationale en el target). Por eso responden directo en inferencia:
  "versión thinking" es el checkpoint base, no el comportamiento entrenado.
- **Ninguna entrenó con rationale**: `rationale_loss_weight = 0.0` en las cuatro
  (las dos corridas previas al soporte de rationale ni siquiera tienen el campo; el
  default del código es `0.0`). La única corrida rationale-first quedó cancelada.

## La matriz de losses (2×2)

La familia de losses se parametriza por `candidate_weighting × candidate_aggregation`.
Las cuatro celdas se entrenaron con datos idénticos, lo que la vuelve una ablation
controlada del lado de entrenamiento.

| Run dir | Loss (`weighting × aggregation`) | Alias | Estado | Eval de creatividad |
|---|---|---|---|---|
| `2026-07-01_1950_qwen3_4b_thinking_10k` | dataset × logsumexp | `concept_set` (default) | ✅ | ✅ (única evaluada) |
| `2026-07-02_2300_qwen3_4b_thinking_10k_softce` | dataset × expected_logprob | `soft_ce` | ✅ | ❌ |
| `2026-07-03_2250_qwen3_4b_thinking_10k_concept_set_uniform` | uniform × logsumexp | `concept_set_uniform` | ✅ | ❌ |
| `2026-07-03_2255_qwen3_4b_thinking_10k_ce_uniform` | uniform × expected_logprob | `ce` | ✅ | ❌ |

Corridas relacionadas que **no** forman parte de la matriz:

- `2026-06-29_1854_qwen3_4b_thinking_10k`: primera prueba, previa a la sugerencia
  de Luciano de hacer el update por receta. **Superada**, no cuenta como celda.
- `2026-07-04_1748_qwen3_4b_rationale_first`: variante con rationale-first,
  **CANCELADA** (nunca terminó).
- `2026-07-03_2105_..._concept_set_uniform`: run dir **vacío** (reintento fallido
  del `concept_set_uniform`; la corrida buena es la de las `2250`).

## Resultado de entrenamiento (dev loss)

Las cuatro corridas convergieron de forma estable (train loss de ~13–17 a ~1.0–1.3,
dev loss monótonamente decreciente hasta el final).

| Variante | dev loss final |
|---|---|
| `concept_set` (dataset × logsumexp) | 1.625 |
| `concept_set_uniform` (uniform × logsumexp) | 1.734 |
| `soft_ce` (dataset × expected_logprob) | 2.054 |
| `ce` (uniform × expected_logprob) | 2.299 |

⚠️ **Estos números NO ordenan a los modelos entre sí.** `logsumexp` y
`expected_logprob` son objetivos distintos y en escalas distintas: el log-sum-exp
sobre un conjunto de candidatos es matemáticamente menor que promediar la NLL por
candidato. Por eso `concept_set` tiene menor loss *por construcción*, no por ser
mejor modelo. La **única** comparación justa es la evaluación generativa río abajo,
que todavía falta correr en 3 de las 4 celdas. La única comparación válida acá es
*weighting vs uniform dentro de la misma agregación*: `dataset` da menor loss que
`uniform` en ambas filas (1.625<1.734 y 2.054<2.299), una señal débil a favor de
ponderar, pero es LM loss, no creatividad.

## Evaluación hecha hasta ahora

Todas las evals de creatividad se corrieron sobre **un solo modelo**, el
`concept_set` (dataset × logsumexp) de `2026-07-01_1950`, con
`src/eval/run_sft_eval.py`. Cuatro carpetas en `eval_outputs/`:

| Carpeta | Alcance | CCS | Notas |
|---|---|---|---|
| `2026-07-01_1950_qwen3_4b_thinking_10k_full_eval` | test completo (1263 recetas), ~40 min | 0.323 | plaus 0.779 · novelty 0.304 · diversity 0.886 |
| `qwen3_4b_l4_eval_100_input_distance` | 100 recetas | 0.304 | tuning de settings de eval |
| `qwen3_4b_l4_eval_100_input_distance_tight` | 100 recetas | 0.313 | idem |
| `qwen3_4b_l4_eval_100_input_distance_tight_v2` | 100 recetas | 0.321 | idem |

Config de eval usada: `num_samples=4`, `max_new_tokens=8`, `temperature=0.6`,
`top_p=0.9`, `top_k=40`, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`,
`alpha=0.8`, `lambda=2.0`, `novelty_method=input_distance`, embeddings
`sentence-transformers/all-MiniLM-L6-v2`.

Fórmula operacional (CCS): `C(x) = α·mean(q^λ·n) + (1−α)·d`, con `q`=plausibilidad,
`n`=novedad (distancia del output a los dos inputs / 2), `d`=diversidad. Con `α=0.8`
y `λ=2`, el CCS queda dominado por el término de novedad, que es bajo (~0.30) y está
acotado ~0.5 por construcción.

### Lectura cualitativa

Los outputs son plausibles pero conservadores y repetitivos: tienden a concatenar
los inputs (`hut+steam → "steamboat cabin hut"`), a repetir la misma muestra
(`cream+ice → "ice cream popper"` ×3) o a derivar (`garden+owl → "nightingale
songbird shelter"`). Esto explica la novedad baja (los outputs quedan cerca de los
inputs) y la plausibilidad decente.

### Baseline base (sin SFT)

El job `2026-07-01 qwen3-4b-base-dev-eval` corrió `src/sft/eval_base.py`, que solo
calcula la **dev LM-loss** del modelo base (no genera, no calcula creatividad) y solo
la imprime en logs (sin archivo). Ese número ya no es recuperable desde los logs.
➡️ **No existe todavía un baseline de creatividad del modelo base**, así que hoy no
se puede afirmar que el SFT mejore la creatividad respecto de Qwen sin entrenar.

> Aclaración de nombres: el job `2026-07-06 qwen3-4b-eval-full-l4-base`, pese al
> "base" en el nombre, re-evaluó el *adapter* `2026-07-01`, no el modelo base.

## Procedencia y reproducibilidad

Cada `run_dir` guarda su procedencia en GCS (persistente, independiente de cualquier laptop):

- `command.txt`: el CLI exacto que se ejecutó.
- `config.yaml`: la config **resuelta** efectiva (todos los flags tras aplicar overrides).
- `data_fingerprint.json`: SHA-256 de `train`/`dev` + versiones de librerías.
- `git_info.json`: commit (ver limitación abajo).

Confirmado con evidencia:

- **Datos idénticos** en las 4 corridas: train `3399f18…` (10.165 líneas), dev
  `cd6c911…` (1.291 líneas). Mismo SHA-256, no solo mismo nombre de archivo.
- **Entorno pinneado**: `transformers 5.12.1`, `torch 2.12.1+cu130`, `peft 0.19.1`,
  `accelerate 1.14.0`, `bitsandbytes 0.49.2`, Python 3.12.13.
- **Config idéntica salvo los ejes de loss**; `length_normalize_concept_logprob: true`
  y `rationale_loss_weight: 0.0` en las 4.

Limitación y su resolución:

- `git_info.json` da `available: false` (git no estaba en el contenedor) y la imagen
  usaba tag `:latest`, así que **no hay SHA de commit por corrida**.
- El único cambio de código relevante en la ventana (07-01 → 07-03) fue `bf3d166`
  (07-02), que refactorizó `losses.py`/`trainer.py`. **Verificado en el diff**: el
  cálculo de la loss de concepto es idéntico (se renombró `causal_concept_logprobs`
  → `causal_masked_logprobs` con la misma matemática; el término de rationale está
  gateado por `rationale_loss_weight > 0`, y las 4 corridas usan `0.0`). ⇒ Las 4
  celdas computaron la **misma** loss de concepto y difieren **solo** en el eje
  `weighting × aggregation`. La ablation es limpia.
- **Recomendación de higiene:** hornear el commit SHA en la imagen (o registrarlo
  desde el trainer) y usar tags inmutables (`:$(git rev-parse --short HEAD)`, ya
  soportado por `cloudbuild.yaml`), para no tener que reconstruir esto a mano nunca más.

## Qué falta y advertencias

1. **La ablation nunca se evaluó río abajo.** 3 de las 4 celdas no tienen ninguna
   métrica generativa. La pregunta central del experimento sigue sin respuesta.
2. **Falta baseline de creatividad del modelo base.** Requiere un modo "sin adapter"
   en `run_sft_eval.py` (hoy siempre carga un adapter PEFT).
3. **No rankear por dev loss** entre losses distintas (ver arriba).
4. **`best_metric` / `best_model_checkpoint` = `None`** en todas las corridas: la
   selección de "mejor adapter" no está atada a una métrica. Fue inocuo (dev loss
   bajó monótonamente, así que best≈final), pero es frágil.
5. **`length_normalize_concept_logprob: true`** quedó activo en todas las corridas
   reales pese a estar marcado como experimental (el default es `false`). Cambia la
   geometría de la loss; conviene decidir si fue intencional.
6. **Higiene de registros:** run dir vacío duplicado (`2105`), job "base" que evaluó
   un adapter. Esta bitácora existe justamente para evitar esa confusión.
7. **Robustez de eval:** `max_new_tokens=8` trunca outputs; muestras repetidas entre
   sí; el tuning de settings de eval se hizo sobre un solo modelo.

## Próximo paso sugerido

No hace falta re-entrenar nada: los adapters ya están. El plan:

1. **Métrica primaria = cobertura/accuracy** contra `known_outputs` (top-1 y top-K
   sobre las 4 muestras; exacta normalizada + opcional semántica por embeddings).
   Es una métrica conocida y de bajo riesgo para elegir la loss. `metrics.py` ya la
   implementa, falta cablearla.
2. **CCS = secundaria, no decisoria todavía.** Se computa en la misma pasada (es casi
   gratis: aritmética sobre embeddings), pero es nuestra contribución en investigación
   y todavía se está validando; no se rankea por ella en esta primera vuelta.
3. **Desacoplar generación de scoring.** La parte cara (GPU) es generar y guardar
   `predictions.jsonl`; una vez guardado, cualquier métrica string/embedding se calcula
   offline en CPU, gratis y repetible. Métricas con LLM-judge sí cuestan aparte.
4. **Correr `run_sft_eval.py` sobre las 4 celdas + baseline base** (`--no_adapter`),
   con una config de eval fija, outputs cortos (~8–10 tokens, la tarea es ≤2 palabras),
   en paralelo. Manejar el `<think>` del modelo base (smoke check para decidir si se
   desactiva el thinking o se recorta post-hoc). Costo: pocos dólares, ~1h wall-clock.
5. Consolidar todo en una tabla comparativa (cobertura + CCS) + una tabla de ejemplos
   cualitativos base vs SFT.

Comparación justa (apples-to-apples): los 4 modelos SFT responden **directo** (fueron
supervisados solo sobre el concepto, sin thinking; se verificó en los outputs guardados),
así que el baseline base también se evalúa en modo respuesta-directa.
