"""Embeddings de texto (sentence-transformers) con cache en disco por hash del string."""
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config


class EmbeddingCache:
    """Cache simple en disco: un .npy por string (hasheado) + índice json."""

    def __init__(self, cache_dir: str = config.CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def get_many(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Devuelve {texto: vector} para los textos que ya están cacheados."""
        found = {}
        for t in texts:
            k = self._key(t)
            if k in self._index:
                path = self.cache_dir / f"{k}.npy"
                if path.exists():
                    found[t] = np.load(path)
        return found

    def set_many(self, texts: list[str], vectors: np.ndarray):
        for t, v in zip(texts, vectors):
            k = self._key(t)
            np.save(self.cache_dir / f"{k}.npy", v)
            self._index[k] = t
        self._save_index()


class Embedder:
    """
    Wrapper sobre sentence-transformers con cache automático.

    Uso:
        embedder = Embedder()
        vectors = embedder.encode(["Fire", "Water", "Steam"])
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        device: str = config.DEVICE,
        use_cache: bool = True,
        cache_dir: str | None = None,
    ):
        # Import diferido para no forzar la dependencia solo al importar el módulo.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        # El cache hashea solo el texto (no el modelo); usar un dir distinto por modelo.
        self.cache = EmbeddingCache(cache_dir or config.CACHE_DIR) if use_cache else None

    def encode(self, texts: Iterable[str], prefix: str = "", show_progress: bool = True) -> np.ndarray:
        texts = list(texts)
        vectors = [None] * len(texts)
        to_compute = []
        to_compute_idx = []

        if self.cache is not None:
            cached = self.cache.get_many(texts)
        else:
            cached = {}

        for i, t in enumerate(texts):
            if t in cached:
                vectors[i] = cached[t]
            else:
                to_compute.append(t)
                to_compute_idx.append(i)

        if to_compute:
            prefixed = [f"{prefix}{t}" for t in to_compute]
            new_vecs = self.model.encode(
                prefixed,
                batch_size=config.BATCH_SIZE,
                show_progress_bar=show_progress,
                normalize_embeddings=True,  # clave: para que cos_sim = dot product
                convert_to_numpy=True,
            )
            for idx, vec in zip(to_compute_idx, new_vecs):
                vectors[idx] = vec
            if self.cache is not None:
                self.cache.set_many(to_compute, new_vecs)

        return np.vstack(vectors)


class FakeEmbedder:
    """Vectores random deterministas por hash del string, para smoke tests (no análisis real)."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: Iterable[str], prefix: str = "", show_progress: bool = False) -> np.ndarray:
        vectors = []
        for t in texts:
            seed = int(hashlib.sha256(t.strip().lower().encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dim)
            v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.vstack(vectors)
