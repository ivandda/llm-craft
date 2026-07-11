# Correcciones a aplicar en el informe LaTeX (`acl_latex.tex`)


Cada dato numérico y cada afirmación metodológica del informe se verificó contra el repo
(configs de entrenamiento, código de `src/`, los `datasets/final-10k/*.jsonl`,
`docs/resultados.md` y `docs/codigo/bitacora_experimentos.md`).

**Veredicto:** el informe es **muy fiel** a lo realizado. Casi todos los números verifican
exactamente. Hay **un (1) error factual**, **un (1) bloqueante** (una tabla sin completar) y
unos pocos arreglos cosméticos. Ninguna de estas correcciones está aplicada — esta es la
lista para que la apliquen ustedes.

---

## 1. Bloqueante

### 1.1 Tabla `tab:embedding_creativity_results` (Apéndice A.1) está entera en `\texttt{TODO}`
Los valores ya están calculados (de los `summary.json` de cada modelo en
`gs://llm-craft-bucket/eval_outputs/cu126_*_test/`; todos con el mismo setting:
backend `sentence_embeddings` (`all-MiniLM-L6-v2`), `α=0.8`, `λ=2`, 4 muestras por receta —
consistente con el resto del informe). **Reemplazar las 6 filas de `TODO` por estos números:**

| Modelo | Plausibilidad `q` | Novedad `n` | Diversidad `d` | Puntaje `C̄` |
|---|---|---|---|---|
| Base | 0.600 | 0.409 | 0.829 | 0.284 |
| Concept-set | 0.780 | 0.304 | 0.887 | 0.324 |
| Concept-set uniforme | 0.776 | 0.306 | 0.877 | 0.322 |
| Soft-CE | 0.786 | 0.303 | 0.850 | 0.318 |
| CE uniforme | 0.781 | 0.301 | 0.841 | 0.314 |
| DPO | 0.818 | 0.284 | 0.112 | 0.173 |

**LaTeX listo para pegar** (reemplaza las filas entre `\midrule` y `\bottomrule` de la tabla):

```latex
Base                 & 0.600 & 0.409 & 0.829 & 0.284 \\
Concept-set          & 0.780 & 0.304 & 0.887 & 0.324 \\
Concept-set uniforme & 0.776 & 0.306 & 0.877 & 0.322 \\
Soft-CE              & 0.786 & 0.303 & 0.850 & 0.318 \\
CE uniforme          & 0.781 & 0.301 & 0.841 & 0.314 \\
DPO                  & 0.818 & 0.284 & 0.112 & 0.173 \\
```

Coherente con el relato del informe: DPO tiene la **mayor** plausibilidad `q` (converge al
canónico) pero la **menor** diversidad `d` (0.112: colapsa a respuestas cortas casi idénticas),
por lo que su puntaje `C̄` cae — otra cara del trade-off correcto/creativo. Este descriptor es
exploratorio (no verifica corrección), tal como ya aclara el texto de esa subsección.

---

## 2. Error factual

### 2.1 `max_new_tokens` de la generación de pares DPO: dice **24**, debería ser **12**
- Aparece en **§3.1** ("Construcción del dataset de preferencias") y en el **Apéndice A.4**
  ("Composición del dataset de preferencias"). En ambos: *"…un máximo de 24 tokens nuevos."*
  → cambiar **24 → 12**.
- **Por qué:** según `bitacora_experimentos.md` (sección "DPO desde soft_ce"), la versión v2
  con `max_new_tokens=24` producía *rejected* de ~17 palabras (*rambles*); la corrida
  **final (v3)** usó **`max_new_tokens=12`** (*rejected* ~8 palabras). El resto de la oración
  (temp 0.9, top-p 0.95, top-k 50, 6 muestras, 2 000 recetas) **es correcto**.

---

## 3. Cosméticos

| # | Ubicación | Corrección |
|---|-----------|-----------|
| 3.1 | §3.1 | `se uso` → `se usó`; `modelo soleccionado` → `modelo seleccionado` |
| 3.2 | Ec. (1) | `\mathcal{Y}(x)={y_1,\ldots,y_m}` → `…=\{y_1,\ldots,y_m\}` (faltan las llaves `\{ \}` del conjunto) |
| 3.3 | Figura | `\includegraphics{Screenshot from 2026-07-10 18-08-47.png}`: el archivo está en `informe/` pero el `.tex` en `informe/latex/`, así que no resuelve al compilar desde `latex/`. Solución: agregar `\graphicspath{{../}{./}}` en el preámbulo, **o** mover/copiar la imagen junto al `.tex`. Recomendado además: renombrarla a algo sin espacios (p. ej. `fig_judge_top1.png`) y, si se puede, reemplazar el screenshot por una figura vectorial/etiquetada. |

---

## 4. Menores / opcionales

- El cuerpo no dice el **split** de los 1 999 pares (**1 799 train / 200 dev**) ni la **tasa
  de DPO** (`5e-5`) ni `β=0.1` en §3.1 — se pueden agregar por reproducibilidad.
- No se menciona `lora_dropout` (0.05 en SFT, 0.0 en DPO). Opcional.

---

## Nota sobre `docs/informe/*.md`
`destilacion_creatividad_composicional.md` y `resumen-repaso.md` son el **anteproyecto**
(propuesta inicial), no descripciones de lo hecho: mencionan
`near_negatives`/`easy_negatives`/`preference_pairs` y métodos (RAG/M5, *goal-directed*,
evaluación humana, TFS) que **no** se ejecutaron. Se les agregó un banner de "documento
superado". El informe fiel es el LaTeX + `docs/resultados.md`.
