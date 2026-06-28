# Pipeline SFT QLoRA

El módulo `src/sft` implementa una pipeline independiente para entrenar un causal LM con LoRA/QLoRA sobre recetas composicionales tipo Infinite Craft.

## Datos

Cada fila JSONL debe incluir `input_a`, `input_b` y una lista `candidate_outputs`:

```json
{
  "input_a": "fire",
  "input_b": "water",
  "candidate_outputs": [
    {"output": "steam", "source": "observed", "rank": 1},
    {"output": "vapor", "source": "teacher", "rank": 2}
  ]
}
```

También se aceptan filas legacy con `outputs` u `output`; se convierten internamente a candidatos.

## Pesos

`--weight_field weight` indica el campo de peso esperado por candidato. Si falta, `--weight_fallback inverse_rank` usa pesos proporcionales a `1/rank`; `uniform` asigna el mismo peso a todos. En todos los casos, los pesos se normalizan por receta para que sumen 1. Si `merge_duplicate_recipes` está activo, filas con el mismo `(input_a, input_b)` se fusionan, se deduplican candidatos por `output` sumando su masa y luego se renormaliza.

## Losses

La loss calcula `log p(candidato | input_a, input_b)` solo sobre los tokens del concepto final y siempre agrupa por receta. El collator expande todos los candidatos aceptables dentro del batch; `per_device_*_batch_size` cuenta recetas, no candidatos tokenizados.

Las dos decisiones ortogonales son:

```text
candidate_weighting   ∈ {uniform, dataset}
candidate_aggregation ∈ {expected_logprob, logsumexp_prob}
```

Con \(\ell_i = \log p_\theta(c_i | x)\) y pesos normalizados \(\alpha_i\):

```text
expected_logprob: -Sum_i alpha_i * ell_i
logsumexp_prob:   -log Sum_i alpha_i * exp(ell_i)
```

Los aliases legacy siguen disponibles:

```text
ce                  -> uniform + expected_logprob
soft_ce             -> dataset + expected_logprob
concept_set         -> dataset + logsumexp_prob
concept_set_uniform -> uniform + logsumexp_prob
```

Si se fijan explícitamente `candidate_weighting` y `candidate_aggregation`, esos valores tienen prioridad práctica sobre el alias. `loss_type` queda como ayuda de compatibilidad para completar ambos ejes cuando no se los configura de forma directa.

`ce_target` se conserva únicamente por compatibilidad hacia atrás. Ya no selecciona un único candidato para CE: el collator siempre expande todos los candidatos aceptables de la receta.

`length_normalize_concept_logprob` mantiene una variante experimental donde la log-probabilidad del concepto se divide por su longitud en tokens antes de agregarse por receta. El comportamiento principal sigue siendo la suma autoregresiva completa del concepto.

## Formato del prompt

```text
Input A: {input_a}
Input B: {input_b}
Final concept: {candidate_output}
```

El collator usa `return_offsets_mapping=True` de tokenizers rápidos para marcar únicamente el span de `{candidate_output}` en `concept_mask`.

## Outputs

Cada ejecución escribe:

```text
runs/sft/<run_id>/
  config.yaml
  command.txt
  git_info.json
  data_fingerprint.json
  metrics.jsonl
  train_losses.jsonl
  eval_losses.jsonl
  plots/
  checkpoints/
  best_adapter/
  final_adapter/
  tokenizer/
  trainer_state.json
```

Los checkpoints incluyen adapter/tokenizer y `accelerate` state para reanudar optimizer, scheduler y RNG.
