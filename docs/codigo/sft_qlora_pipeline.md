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
Si cada candidato trae `rationale`, la pipeline puede activar un término auxiliar de entrenamiento
con `rationale_loss_weight > 0`.

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

No se permiten configuraciones parciales: si se define uno de los dos ejes explícitos, se deben definir ambos. Si no se define ninguno, ambos se derivan automáticamente desde `loss_type`.

Si tampoco se define `loss_type`, se usan los defaults de `SFTConfig`, por lo que la configuración efectiva queda en `concept_set`, equivalente a `candidate_weighting="dataset"` y `candidate_aggregation="logsumexp_prob"`.

`ce_target` se conserva únicamente por compatibilidad hacia atrás. Ya no selecciona un único candidato para CE: el collator siempre expande todos los candidatos aceptables de la receta.

`length_normalize_concept_logprob` mantiene una variante experimental donde la log-probabilidad del concepto se divide por su longitud en tokens antes de agregarse por receta. El comportamiento principal sigue siendo la suma autoregresiva completa del concepto.

### Término auxiliar de rationales

Para el experimento M3, activar:

```yaml
rationale_loss_weight: 0.2
length_normalize_rationale_logprob: true
rationale_position: output_before_rationale
```

La loss total queda:

```text
total_loss = concept_loss + rationale_loss_weight * rationale_loss
```

`concept_loss` conserva exactamente el esquema multi-candidato configurado por
`candidate_weighting` y `candidate_aggregation`. `rationale_loss` usa el mismo esquema, pero
solo sobre los tokens del campo `rationale`. Si un candidato no tiene rationale, sigue aportando
a `concept_loss` y no aporta al término auxiliar.

`rationale_position` permite comparar dos órdenes autoregresivos sin cambiar el dataset ni la
definición de la loss:

```text
output_before_rationale  # concepto primero, explicación después
output_after_rationale   # explicación primero, concepto final después
```

## Formato del prompt

```text
Input A: {input_a}
Input B: {input_b}
Final concept: {candidate_output}
```

El collator usa `return_offsets_mapping=True` de tokenizers rápidos para marcar únicamente el span de `{candidate_output}` en `concept_mask`.

Opcionalmente, `prompt_format: qwen_chat` renderiza un prompt estilo instrucción dentro del `chat template` del tokenizer. Esto está pensado para variantes Qwen chat/thinking: el usuario recibe el bloque

```text
Given two concepts, combine them into one resulting concept.

Concept A: {input_a}
Concept B: {input_b}

Return only the resulting concept.
```

y el candidato supervisado se inserta como contenido del mensaje `assistant`. El span supervisado sigue siendo solo el concepto final, no los tokens estructurales del template ni trazas de razonamiento.

Con `rationale_loss_weight > 0`, el texto supervisado agrega:

```text
Rationale: {rationale}
```

Con `rationale_position: output_before_rationale`, el ejemplo plain completo es:

```text
Input A: fire
Input B: water
Final concept: steam
Rationale: Fire heats water until it becomes steam.
```

Con `rationale_position: output_after_rationale`, el ejemplo plain completo es:

```text
Input A: fire
Input B: water
Rationale: Fire heats water until it becomes steam.
Final concept: steam
```

En `prompt_format: qwen_chat`, el mismo orden se aplica dentro del mensaje `assistant`:

```text
steam
Rationale: Fire heats water until it becomes steam.
```

o bien:

```text
Rationale: Fire heats water until it becomes steam.
Final concept: steam
```

El concepto final y el rationale tienen máscaras separadas, por lo que el peso auxiliar no cambia
la definición de `concept_loss`.

Comandos recomendados:

```bash
# Concepto y luego rationale
uv run python -m src.sft.train \
  --config configs/sft/qwen05b_10k_rationale_example.yaml

# Rationale y luego concepto
uv run python -m src.sft.train \
  --config configs/sft/qwen05b_10k_rationale_first_example.yaml
```

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
