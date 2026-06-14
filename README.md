# llm-craft: Destilación de Creatividad Composicional

Este repositorio contiene las herramientas y el pipeline de datos para el proyecto de destilación y evaluación de creatividad composicional en LLMs pequeños, inspirado en el juego *Infinite Craft*.

---

## Configuración Inicial

Para instalar las dependencias y configurar el entorno virtual utilizando `uv`:

```bash
# Sincronizar e instalar dependencias
uv sync
```

---

## Pipeline de Datos

El pipeline se ejecuta secuencialmente para procesar los datasets crudos, limpiarlos, generar particiones y exportar los formatos de entrenamiento y evaluación.

### 1. Normalización de Datos
Carga las distintas fuentes de datos crudos (`datasets/raw/`) y unifica sus formatos en un único archivo de observaciones.
```bash
uv run python -m src.data.normalize
```
* **Qué hace**: Parsea JSONL, CSV y estructuras complejas de Infinite Craft, propaga y unifica los emojis a nivel case-insensitive, ordena alfabéticamente los ingredientes y escribe en `recipe_observations_v0.jsonl`.

### 2. Limpieza, Agregación y Particiones (Splits)
Deduplica las observaciones raw, calcula conflictos, y asigna splits deterministas.
```bash
uv run python -m src.data.clean
```
* **Qué hace**: Resuelve la capitalización de cada concepto al caso más frecuente (mode casing), identifica combinaciones con múltiples respuestas válidas (`is_conflicting_pair`), descarta errores y asigna cada combinación a `train`/`dev`/`test` usando hashes del par de entrada para evitar filtración de datos (leakage). Genera `recipe_canonical_v0.jsonl` y `clean_metrics.json`.

### 3. Exportación para Fine-Tuning (SFT)
Prepara los datasets conversacionales para el entrenamiento supervisado del modelo.
```bash
uv run python -m src.data.export_sft
```
* **Qué hace**: Exporta dos variantes conversacionales en formato `messages`: `sft_clean` (recetas sin conflictos ni copias triviales) y `sft_all` (todas las recetas válidas), incluyendo metadatos enriquecidos (`pair_id`, `recipe_id`, etc.).

### 4. Exportación de Conjuntos de Evaluación
Genera los datasets estructurados para evaluar la calidad y creatividad de las respuestas generadas.
```bash
uv run python -m src.data.export_eval
```
* **Qué hace**: Utiliza un algoritmo de *Reservoir Sampling* de alta eficiencia (bajo consumo de memoria) para extraer muestras deterministas para evaluación (`eval_dev_1k`, `eval_test_1k`, etc.) que listan todas las respuestas válidas conocidas por combinación.

---

## Documentación del Proyecto

Para más detalles teóricos y de diseño, consulte:
* [data_pipeline.md](docs/data_pipeline.md): Especificaciones técnicas de la limpieza, hashes y formato SFT.
* [data_normalization.md](docs/data_normalization.md): Proceso de extracción inicial de datasets crudos.
* [destilacion_creatividad_composicional.md](docs/destilacion_creatividad_composicional.md): Paper de diseño del proyecto de investigación.