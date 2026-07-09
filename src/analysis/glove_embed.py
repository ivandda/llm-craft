"""Backend de embeddings estáticos word-level (GloVe), interfaz compatible con Embedder.

Conceptos multi-palabra: mean-pool de los vectores de sus tokens. Tokens OOV se
ignoran; si todos los tokens de un concepto son OOV, el vector queda en ceros
(contado en n_oov_concepts). Salida L2-normalizada.
"""
import re
from typing import Iterable

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class GloveEmbedder:
    def __init__(self, path: str = config.GLOVE_PATH, dim: int | None = None):
        self.path = path
        self.dim = dim
        self._vec: dict[str, np.ndarray] | None = None
        # estadísticas de cobertura, se acumulan entre llamadas a encode()
        self.n_concepts = 0
        self.n_oov_concepts = 0  # conceptos sin NINGÚN token en vocab
        self.n_tokens = 0
        self.n_oov_tokens = 0

    def _load(self):
        """Carga el .txt de GloVe una sola vez a un dict palabra->vector."""
        if self._vec is not None:
            return
        vec: dict[str, np.ndarray] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) < 3:
                    continue
                vec[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
        if not vec:
            raise RuntimeError(f"GloVe vacío o ilegible en {self.path}")
        self._vec = vec
        self.dim = len(next(iter(vec.values())))

    def _embed_one(self, text: str) -> np.ndarray:
        assert self._vec is not None
        toks = _TOKEN_RE.findall(text.lower())
        vecs = []
        for t in toks:
            self.n_tokens += 1
            v = self._vec.get(t)
            if v is None:
                self.n_oov_tokens += 1
            else:
                vecs.append(v)
        self.n_concepts += 1
        if not vecs:
            self.n_oov_concepts += 1
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def encode(self, texts: Iterable[str], prefix: str = "", show_progress: bool = False) -> np.ndarray:
        # prefix se ignora (no aplica a embeddings estáticos).
        self._load()
        out = np.vstack([self._embed_one(t) for t in texts])
        # L2-normalizar (para que dot = coseno, igual que el Embedder contextual).
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def oov_report(self) -> str:
        ct = 100 * self.n_oov_tokens / max(1, self.n_tokens)
        cc = 100 * self.n_oov_concepts / max(1, self.n_concepts)
        return (f"GloVe OOV: {self.n_oov_tokens}/{self.n_tokens} tokens ({ct:.1f}%), "
                f"{self.n_oov_concepts}/{self.n_concepts} conceptos sin ningún token ({cc:.1f}%)")
