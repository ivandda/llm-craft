# Pipeline de evaluación de creatividad

`src/eval/run_sft_eval.py` agrega una evaluación generativa para adapters SFT y soporta dos formatos de input:

* `datasets/processed/eval_*.jsonl`
* datasets multi-output tipo `datasets/final-10k/test.jsonl`

## Qué calcula

Por cada ejemplo:

1. genera `K` muestras del modelo,
2. toma la primera como `prediction` principal,
3. calcula los componentes de creatividad del survey:
   `plausibility`, `novelty`, `diversity`, `local_creativity`.

Por default, `summary.json` reporta solo creatividad y además incluye las recetas con `local_creativity` mínima y máxima, junto con sus `sampled_outputs`.

## Operacionalización actual

El survey deja abierta la elección del juez de plausibilidad y del encoder semántico. Para mantener la evaluación offline y reproducible en este repo:

* la plausibilidad se reporta como **distancia coseno entre cada muestra y el embedding promedio de los candidatos conocidos** de la receta,
* la fórmula usa `1 - distancia / 2` como score de plausibilidad, así queda acotado en `[0, 1]`,
* la novedad se obtiene con un **judge LLM** en Vertex AI usando un modelo partner no-Google,
* y la diversidad se calcula como la **distancia coseno promedio entre muestras**, pero se transforma a score con `1 - distancia / 2`.

La implementación actual usa Anthropic Claude en Vertex AI para novedad. Para embeddings, el evaluador soporta tres backends:

* `glove`
* `word2vec`
* `sentence_embeddings`

`glove` y `word2vec` cargan archivos de embeddings en texto plano usando promedio de tokens para conceptos multi-palabra. `sentence_embeddings` usa un modelo de Sentence Transformers.

Si `--novelty_method vertex_judge`, el evaluador hace **una sola llamada al judge por receta** y le pasa juntas todas las muestras generadas para ese input. Eso reduce bastante el número de requests respecto de evaluar cada muestra por separado.

Ejemplo:

```bash
uv run --group vertex python -m src.eval.run_sft_eval \
  --run_dir runs/sft/2026-06-28_1213_ce_uniform \
  --eval_file datasets/final-10k/test.jsonl \
  --max_examples 250 \
  --num_samples 3 \
  --embedding_backend sentence_embeddings \
  --novelty_judge_model claude-3-5-sonnet-v2@20241022
```

`--max_examples` te permite evaluar solo las primeras `N` recetas del dataset de test.
