"""Configuración del paquete de análisis."""
import os

# --- Modelo de embeddings ---
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_QUERY_PREFIX = ""  # ej: "query: " para e5
EMBEDDING_PASSAGE_PREFIX = ""  # ej: "passage: " para e5


def _default_device() -> str:
    """GPU si hay CUDA, si no CPU. Se puede forzar con ANALYSIS_DEVICE=cuda|cpu."""
    forced = os.environ.get("ANALYSIS_DEVICE")
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _default_device()  # "cuda" si hay GPU disponible, "cpu" si no
BATCH_SIZE = 64

# --- Clustering ---
RELATION_POOLING = "diff_mean"  # r = emb_c - (emb_a + emb_b) / 2
KMEANS_K_RANGE = range(4, 41)
CLUSTERING_METHOD = "hdbscan"  # "kmeans" o "hdbscan" (marca outliers como -1)
HDBSCAN_MIN_CLUSTER_SIZE = 8
HDBSCAN_MIN_SAMPLES = 3

# --- Novedad (k-NN) ---
NOVELTY_K_NEIGHBORS = 10

# --- Rutas (gitignoreadas, ver .gitignore) ---
CACHE_DIR = os.environ.get("ANALYSIS_CACHE_DIR", "runs/analysis/embedding_cache")
GLOVE_PATH = os.environ.get(
    "ANALYSIS_GLOVE_PATH", "src/analysis/embeddings/glove/glove.6B.300d.txt"
)
