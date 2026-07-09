# clustering — reglas por vector de relación o racionalización

Testea si las reglas de combinación del juego (`a + b -> c`) se pueden descubrir
sin supervisión clusterizando embeddings, con dos representaciones: el vector de
relación `r = emb(c) - (emb(a)+emb(b))/2` (DIFFVEC, Vylomova et al. 2016; SeVeN,
Espinosa-Anke y Schockaert 2018), y los embeddings del texto de la racionalización
de cada tripleta.

## Resultados

El vector de relación no forma clusters. Con GloVe (estáticos word-level, la
familia para la que se diseñó la aritmética de analogías) el silhouette queda en
0.010 (`diff_mean`) y 0.013 (`diff_concat`). Se sostiene cambiando el algoritmo
(KMeans y HDBSCAN, que fragmenta en 737 clusters con ~65% de outliers) y el
pooling. La regla composicional no es un offset lineal consistente.

Clusterizar las racionalizaciones agrupa por dominio, no por tipo de regla. Con
embeddings de oración (bge) sobre el texto del `rationale` el silhouette sigue
bajo (~0.03), pero los grupos son temáticamente coherentes (mezcla geológica,
combustión, crecimiento vegetal, cocina, etc.). Recuperar tipos de regla
(fusión / profesión / wordplay) requeriría etiquetado por un LLM.

## Reproducir los números del informe

Deps opt-in: `uv sync --group analysis`. Todo se corre desde la raíz del repo.

Vector de relación (silhouette 0.010 / 0.013, GloVe) — mirar el bloque REL:

    uv run python -m src.analysis.clustering.validate \
        --train_jsonl datasets/final-10k/train.jsonl --model glove

`--model glove` descarga los vectores GloVe a `src/analysis/embeddings/`.

Racionalizaciones (silhouette ~0.03 y ejemplos por cluster, bge):

    uv run python -m src.analysis.clustering.inspect_rationale_clusters --k 10

El script reutiliza `runs/analysis/rationale_emb.npy` si existe; si no, recomputa los embeddings.

## Estructura

Infra compartida en `src/analysis/`: `config.py`, `embed_utils.py`,
`candidates_loader.py`, `embedder_registry.py`, `glove_embed.py`.

En `clustering/`:

- `relation_vectors.py` — poolings `r = f(a, b, c)`
- `validate.py` — diagnóstico escalable (silhouette del vector de relación y de las racionalizaciones)
- `inspect_rationale_clusters.py` — clusters de racionalizaciones con ejemplos centrales
