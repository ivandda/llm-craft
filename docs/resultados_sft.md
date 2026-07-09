# Resultados de la evaluación de SFT

**Alcance:** evaluación generativa de las 4 variantes de loss + el modelo base sobre el
test completo (**1263 recetas**) de `datasets/final-10k`. **Fecha:** 2026-07-08.

Documento de **resultados**: qué se evaluó, con qué métricas (cada una explicada, con sus
límites) y qué salió. El historial cronológico y los detalles de infraestructura están en
`docs/codigo/bitacora_experimentos.md`.

---

## 1. Resumen ejecutivo

- **El SFT habilita la tarea.** El modelo base (sin adapter) saca **0%** en toda métrica de
  cobertura: no produce conceptos sino fragmentos de razonamiento. El SFT lo vuelve on-task.
- **Ganó `soft_ce` (`dataset × expected_logprob`).** Lidera cobertura exacta, semántica y
  brevedad. El eje que decide es la **agregación de candidatos** (`expected_logprob` ≈ 2×
  `logsumexp`); el *weighting* (dataset vs uniforme) es de segundo orden.
- **El modelo sabe *qué* responder, pero no *cómo* de corto.** La cobertura exacta es baja
  (7%) pero un juez LLM lo encuentra **correcto el 83% de las veces**: el conocimiento
  composicional está; el cuello de botella es la **forma** (verbosidad, tokens cortados).
- **Más de la mitad de sus respuestas son descubrimientos** (correctos **y** fuera de la
  lista del teacher): 53.6% top-1. El alumno generaliza, no memoriza.

---

## 2. Qué se evaluó

### 2.1 Modelos

| Etiqueta | loss (`weighting × aggregation`) | `run_dir` |
|---|---|---|
| `concept_set` | dataset × logsumexp | `2026-07-01_1950_qwen3_4b_thinking_10k` |
| `soft_ce` **(ganador)** | dataset × expected_logprob | `2026-07-02_2300_..._softce` |
| `concept_set_uniform` | uniform × logsumexp | `2026-07-03_2250_..._concept_set_uniform` |
| `ce_uniform` | uniform × expected_logprob | `2026-07-03_2255_..._ce_uniform` |
| `base` | — (sin adapter) | `Qwen/Qwen3-4B-Thinking-2507` crudo |

Base: `Qwen/Qwen3-4B-Thinking-2507` en 4-bit (QLoRA). Los 4 adapters comparten datos y
config, y difieren **solo** en el eje de loss (ablation controlada).

### 2.2 Datos y generación

- **Test:** `final-10k/test.jsonl`, 1263 recetas. Splits por hash del par ⇒ ningún par de
  test aparece en train/dev. Cada receta trae `known_outputs` (salidas plausibles del
  teacher) y `canonical_output`.
- **Generación (idéntica en las 5 evals):** `num_samples=4`, `max_new_tokens=8`,
  `temperature=0.6`, `top_p=0.9`, `top_k=40`, `repetition_penalty=1.15`,
  `no_repeat_ngram_size=3`, `max_concept_words=3`. Prompt del student = el simple de
  `collator.py`, igual en train y eval.
- **Base:** evaluado con `--close_think_prompt` + `--strip_think` (el trato más favorable
  para forzarlo a responder directo, sin el `<think>` que inyecta siempre).
- **Generación (GPU) y scoring (CPU/juez) están desacoplados:** una vez guardado
  `predictions.jsonl`, cualquier métrica se recalcula sin re-generar.

---

## 3. Métricas (qué mide cada una, pros y contras)

Hay **dos familias**: **cobertura** ("¿acertó a la respuesta esperada?") y **creatividad**
("¿es correcta y nueva?"). Se leen juntas; ninguna sola alcanza.

### 3.1 Cobertura exacta *(primaria, decisoria)*
Fracción de recetas donde la muestra coincide *string a string* (normalizado) con algún
`known_output` (`known`) o con el `canonical`. `top1` = primera muestra; `any@k` = alguna de las 4.
- **Pros:** objetiva, determinista, gratis, 100% reproducible. Ideal para elegir variante.
- **Contras:** brutalmente estricta — castiga sinónimos (`fax machine` ≠ `faxmachine`),
  verbosidad (`lightning rod system` ≠ `lightning`) y respuestas correctas fuera de la
  lista. Subestima el desempeño real. Depende de la calidad de `known_outputs`.

### 3.2 Cobertura semántica *(secundaria, robustez)*
Igual, pero una muestra acierta si su embedding (`all-MiniLM-L6-v2`) tiene coseno ≥ 0.75
con algún `known_output`. Rescata sinónimos y paráfrasis.
- **Pros:** robusta a la forma, barata, determinista.
- **Contras:** umbral arbitrario; los embeddings confunden *relacionado* con *correcto*
  (`fire`↔`firefighter`). No verifica correctitud lógica, solo cercanía.

### 3.3 Verbosidad *(diagnóstico de forma)*
Media de palabras del top-1 y fracción de ≤2 palabras (la tarea apunta a ≤2).
- **Pros:** diagnóstico directo del problema de sobre-especificación; trivial e interpretable.
- **Contras:** no es calidad (corto puede ser incorrecto); se lee junto a cobertura, no sola.

### 3.4 CCS de embeddings *(exploratoria — es un PROXY, no un juicio)*
`C = α·mean(qᵏ·n) + (1−α)·d`, α=0.8, λ=2. Los tres componentes son **distancias coseno**,
sin ningún modelo que juzgue correctitud:
- **plausibility (q):** cercanía de la muestra al *centroide de los `known_outputs`*.
- **novelty (n):** distancia de la muestra a los dos *inputs* (NO respecto al dataset).
- **diversity (d):** dispersión entre las 4 muestras.
- **Pros:** barato, señal continua de forma/variedad; chequeo de cordura.
- **Contras:** **no verifica correctitud** (premió al base por "novedad" cuando escupía
  `"Hmm the user"`); `novelty` mide distancia al input, no creatividad; su valor absoluto
  es poco interpretable. ⇒ Descriptor, no veredicto.

### 3.5 Juez LLM — Validez y Creatividad *(la métrica "de verdad")*
Un LLM **más fuerte que el teacher** juzga, **sin ver los `known_outputs`**, si la salida
es un resultado *plausible* de combinar A+B. Se cruza con una comprobación determinista de
pertenencia al dataset (exacto o coseno ≥ 0.75). Definición de creatividad = **valor ×
novedad** (Boden): la novedad solo cuenta si además es válida.

Cubetas por salida → dos tasas independientes (top-1 y any@k):
```
invalid      = no plausible                 -> mal / malformado
valid-known  = plausible Y en-dataset       -> correcto, pero no nuevo
valid-novel  = plausible Y fuera-de-dataset -> DESCUBRIMIENTO (correcto + nuevo)

Validez    = (valid-known + valid-novel) / N   -> con qué frecuencia acierta
Creatividad =  valid-novel / N                 -> con qué frecuencia acierta Y es nuevo
```
- **Por qué sin referencia:** para que el juez no se limite a sellar coincidencias ni
  castigue respuestas válidas no listadas — así acredita cuando el alumno supera al teacher.
- **Por qué dos tasas separadas:** no colapsar "correcto" con "creativo"; se ve el trade-off.
- **Pros:** es lo más cercano a un juicio humano; ve lo que la cobertura no.
- **Contras:** cuesta y es no-determinista (temp=0 mitiga); depende del juez y el prompt;
  es **lene con la verbosidad** (plausibilidad ≠ concisión) — por eso brevedad va aparte.
  Un solo juez, sin intervalos de confianza (ver §7).
- **Juez usado:** **Gemini 2.5 Pro** en batch. *(Claude Sonnet era la 1ª opción por ser
  proveedor independiente del teacher, pero la cuenta de créditos no puede habilitarlo en
  Marketplace; detalle en la bitácora.)*

### 3.6 Cómo leerlas juntas

| Pregunta | Métrica | Confianza |
|---|---|---|
| ¿Qué variante elijo? | Cobertura exacta/semántica | alta (decisoria) |
| ¿Responde en formato corto? | Verbosidad | alta (forma) |
| ¿Es correcta de verdad? | **Validez (juez)** | alta |
| ¿Es correcta **y** nueva? | **Creatividad (juez)** | alta |
| Descriptor de forma/variedad | CCS embeddings | baja (exploratoria) |

---

## 4. Resultados

### 4.1 Cobertura (exacta y semántica)

| Modelo | top1_known | any@k_known | top1_canon | top1_sem | any@k_sem |
|---|---|---|---|---|---|
| **`soft_ce`** | 0.073 | **0.143** | **0.042** | **0.337** | **0.511** |
| `ce_uniform` | **0.075** | 0.134 | 0.038 | 0.290 | 0.476 |
| `concept_set` | 0.040 | 0.077 | 0.026 | 0.274 | 0.410 |
| `concept_set_uniform` | 0.038 | 0.075 | 0.025 | 0.259 | 0.394 |
| **base** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 4.2 Verbosidad

| Modelo | media palabras (top-1) | ≤2 palabras |
|---|---|---|
| **`soft_ce`** | **2.50** | **40.9%** |
| `ce_uniform` | 2.59 | 33.7% |
| `concept_set` | 2.71 | 24.0% |
| `concept_set_uniform` | 2.78 | 18.2% |
| base | 2.99 | 1.3% |

### 4.3 CCS de embeddings *(proxy — ver §3.4)*

> 🐛 **Corrección de un bug de signo en `diversity`.** El término `diversity_score` estaba
> como `1 − d/2` (premiaba que las muestras fueran *parecidas*, lo opuesto a "diversidad").
> Se corrigió a `d/2` (alto = más variado) en `creativity.py`. La tabla muestra los valores
> **corregidos**, recalculados por-receta desde los `predictions.jsonl` guardados (exacto,
> sin re-generar). Los `summary.json` históricos en GCS tienen los valores viejos (buggy).

| Modelo | CCS (corregido) | plaus | novelty | diversity |
|---|---|---|---|---|
| **`soft_ce`** | **0.178** | 0.786 | 0.303 | 0.150 |
| `ce_uniform` | 0.177 | 0.781 | 0.301 | 0.159 |
| `concept_set_uniform` | 0.171 | 0.776 | 0.306 | 0.123 |
| `concept_set` | 0.169 | 0.780 | 0.304 | 0.113 |
| base | 0.152 | 0.600 | 0.409 | 0.171 |

Con la diversidad bien orientada, **el CCS ahora coincide con cobertura**: las variantes de
`expected_logprob` (`soft_ce`, `ce_uniform`) lideran. La conclusión previa "el CCS
contradice a cobertura" era **un artefacto del bug**. Aun así la banda entre las 4 celdas
(0.169–0.178) es angosta, así que el CCS sigue siendo un descriptor débil, no un criterio:
el veredicto de calidad lo da el juez (§4.4). El `base` queda último pese a su alta novedad
y diversidad, porque su baja plausibilidad (elevada al cuadrado, λ=2) lo hunde.

### 4.4 Juez LLM — Validez y Creatividad

**Solo se juzgó `soft_ce`** (el ganador por cobertura). Gemini 2.5 Pro, batch, 1263 recetas:

| Métrica | Juez LLM | Cobertura exacta | Cobertura semántica |
|---|---|---|---|
| **Validez** top-1 (correcto) | **0.833** | 0.073 | 0.337 |
| **Validez** any@k | **0.968** | 0.143 | 0.511 |
| **Creatividad** top-1 (correcto **y** nuevo) | **0.536** | — | — |
| **Creatividad** any@k | **0.769** | — | — |

Cubetas del top-1: **inválido 16.7% · válido-conocido 29.7% · descubrimiento 53.6%**.

1. **El modelo acierta el 83% top-1** según el juez — ~11× el exact-match (7.3%). El
   exact-match subcontaba por verbosidad, sinónimos y respuestas correctas fuera de lista.
2. **53.6% de los top-1 son descubrimientos** (correctos y fuera del dataset): el alumno
   generaliza (`castle wall+castle wall → castle fortress complex`, `algae+algae → biofilm`).
3. **Con 4 muestras casi siempre hay una válida** (any@k 96.8%; 76.9% con descubrimiento).
   Refuerza el valor de muestrear K y de un DPO que empuje la buena muestra al top-1.
4. **El 16.7% inválido** (tokens cortados, `'owl garden a'`, comillas sueltas) es el techo
   de mejora inmediato — atacable con decoding y/o DPO.

Datos: `gs://llm-craft-bucket/eval_outputs/judge_softce_gemini/{summary,judgments}.jsonl`.

---

## 5. Análisis cualitativo (`soft_ce`)

Sobre las 1263 recetas (top-1): largo 1/2/3 palabras = 9%/32%/59%; colapso de modo (4
muestras idénticas) 13.6%; eco de input 43.2%; top-1 distintos 96.4%.

**El modelo entiende** — muchos "fallos" de exact-match son correctos:

| Receta | known | predicción | juicio |
|---|---|---|---|
| element + skull | calcium, bone | calcium phosphate | correcto (¡mejor que el label!) |
| paper + phone | faxmachine | fax machine | exacto (solo el espacio) |
| earth + fire | lava, magma | lava rock | correcto |
| land + organic matter | soil, compost | soil fertility | correcto |
| energy + tree | photosynthesis | **photosynthesis** | exacto |

**Problemas de forma (no de comprensión):**
1. **Verbosidad / sobre-especificación** (el #1): "lightning rod system" en vez de
   "lightning". El 59% son de 3 palabras; penaliza directo el exact-match.
2. **Tokens corruptos:** `snow globe orbiterator`, `owl garden a` — artefactos del corte a
   8 tokens y/o del `repetition_penalty`.
3. **Colapso de modo (13.6%):** 4 muestras idénticas; reduce el beneficio de `any@k`.

**Base vs SFT:** con el mejor trato del thinking, el base produce meta-texto (`"Hmm the
user"`, `"We are combining"`), no conceptos. El SFT no "mejora" al base: lo **habilita**.

---

## 6. Veredicto: ¿es bueno el modelo?

**Como motor de asociación composicional: sí, sorprendentemente.** Entiende cómo se
combinan dos conceptos y produce resultados válidos (83% según el juez), a veces mejores
que la referencia, sobre 1263 pares no vistos, con 96% de top-1 distintos.

**Como generador del formato que la tarea premia: todavía no.** La cobertura exacta (7%) es
baja porque castiga verbosidad y sinónimos; sumado a tokens corruptos y colapso de modo,
la "calidad de producto" falta.

**En una frase:** el SFT resolvió lo difícil (el conocimiento) y dejó lo fácil-pero-clave
(brevedad y formato). La brecha exacto↔juez dice dónde invertir: **no en más razonamiento,
sino en control de forma.**

---

## 7. Limitaciones

- **Creatividad medida solo en el ganador** (`soft_ce`). No hay comparación de creatividad
  entre variantes: para eso habría que juzgar al menos `concept_set` (el mejor `logsumexp`)
  y ver si el eje de agregación también domina en la métrica del juez.
- **Un solo juez, sin intervalos de confianza.** Gemini 2.5 Pro es de la misma familia que
  el teacher (2.5 Flash); un 2º juez o un chequeo humano sobre ~30 casos daría más rigor.
  El juez es lene con la verbosidad (por eso brevedad va aparte).
- **Labels ruidosos:** algunos `known_outputs` del teacher son raros (`bones+rope →
  "needleandthread"`), así que el exact-match subestima aún más.
- **`max_new_tokens=8`** probablemente causa parte de los tokens cortados; no se probó mayor.
- **Diferencias chicas dentro del ruido:** la conclusión fuerte es *agregación >> weighting*
  y *SFT >> base*, no el orden fino entre las dos mejores.

---

## 8. Próximos pasos

1. **(Opcional) Juzgar `concept_set`** para cerrar la comparación de creatividad en el eje
   que importa (agregación). ~$7 en batch.
2. **DPO desde `soft_ce`** (alto impacto). Preferidos = `known_outputs` cortos/canónicos;
   rechazados = las muestras verbosas/corruptas del propio modelo. Ataca de frente brevedad
   + correctitud, y el any@k 96.8% dice que la buena muestra ya existe: falta rankearla top-1.
3. **Re-decodificar el ganador** (`max_new_tokens≈12`, `repetition_penalty` más suave) para
   ver cuánto del 16.7% inválido es artefacto de decoding. Barato, sin reentrenar.
4. **Higiene de datos:** recortar `known_outputs` a ≤2 palabras para alinear target y métrica.

---

## 9. Apéndice — reproducción

**Predicciones:** `gs://llm-craft-bucket/eval_outputs/cu126_<variante>_test/predictions.jsonl`
(`<variante> ∈ {concept_set, softce, concept_set_uniform, ce_uniform, base}`).

```bash
# Cobertura (offline, CPU, gratis) — las 5 variantes de una:
OUT=gs://llm-craft-bucket/eval_outputs
uv run python -m src.eval.score_coverage \
  $OUT/cu126_{concept_set,softce,concept_set_uniform,ce_uniform,base}_test/predictions.jsonl \
  --labels concept_set,soft_ce,concept_set_uniform,ce_uniform,base --semantic_threshold 0.75

# Juez de creatividad (Gemini 2.5 Pro, batch):
uv run --group vertex python -m src.eval.judge_creativity \
  $OUT/cu126_softce_test/predictions.jsonl \
  --model gemini-2.5-pro --mode batch \
  --gcs_staging gs://llm-craft-bucket/judge_batch --semantic_threshold 0.75 \
  --output_dir $OUT/judge_softce_gemini
```
