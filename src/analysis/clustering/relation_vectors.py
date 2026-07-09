"""Poolings para el vector de relación r = f(emb_a, emb_b, emb_c) (DIFFVEC, SeVeN)."""
import numpy as np


def diff_mean_pooling(emb_a: np.ndarray, emb_b: np.ndarray, emb_c: np.ndarray) -> np.ndarray:
    """r = c - (a + b) / 2."""
    return emb_c - (emb_a + emb_b) / 2.0


def concat_pooling(emb_a: np.ndarray, emb_b: np.ndarray, emb_c: np.ndarray) -> np.ndarray:
    """r = [a, b, c] concatenados (para un clasificador supervisado sobre r)."""
    return np.concatenate([emb_a, emb_b, emb_c], axis=-1)


def diff_concat_pooling(emb_a: np.ndarray, emb_b: np.ndarray, emb_c: np.ndarray) -> np.ndarray:
    """r = [c - a, c - b]; preserva la relación de c con cada input por separado."""
    return np.concatenate([emb_c - emb_a, emb_c - emb_b], axis=-1)


POOLING_STRATEGIES = {
    "diff_mean": diff_mean_pooling,
    "concat": concat_pooling,
    "diff_concat": diff_concat_pooling,
}


def compute_relation_vectors(
    emb_a: np.ndarray, emb_b: np.ndarray, emb_c: np.ndarray, strategy: str = "diff_mean"
) -> np.ndarray:
    """Aplica el pooling elegido a arrays alineados por fila (a_i, b_i, c_i)."""
    if strategy not in POOLING_STRATEGIES:
        raise ValueError(f"Estrategia desconocida: {strategy}. Opciones: {list(POOLING_STRATEGIES)}")
    return POOLING_STRATEGIES[strategy](emb_a, emb_b, emb_c)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normaliza cada fila a norma 1 (para clustering por coseno)."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms
