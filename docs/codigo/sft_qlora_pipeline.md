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

`--weight_field weight` indica el campo de peso esperado por candidato. Si falta, `--weight_fallback inverse_rank` usa pesos proporcionales a `1/rank`; `uniform` asigna el mismo peso a todos. En todos los casos, los pesos se normalizan por receta para que sumen 1.

## Losses

`concept_set` aplana candidatos en el collator, calcula `log p(candidato | input_a, input_b)` solo sobre los tokens del concepto final y agrupa por receta:

```text
-logsumexp_i(log w_i + log p(c_i | x))
```

`ce` selecciona un único candidato por receta con `--ce_target rank1`, `observed` o `first`, y aplica la CE causal solo sobre los tokens del concepto.

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
