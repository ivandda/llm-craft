from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class BaseTextEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError


class StaticWordVectorEmbedder(BaseTextEmbedder):
    def __init__(self, vectors: dict[str, torch.Tensor], *, dimensions: int) -> None:
        self.vectors = vectors
        self.dimensions = dimensions
        self.cache: dict[str, torch.Tensor] = {}

    @classmethod
    def from_text_file(cls, path: str | Path) -> "StaticWordVectorEmbedder":
        vectors: dict[str, torch.Tensor] = {}
        dimensions: int | None = None
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if line_number == 1 and len(parts) == 2 and all(part.isdigit() for part in parts):
                    continue
                if len(parts) < 2:
                    continue
                word = parts[0].lower()
                try:
                    vector = torch.tensor([float(value) for value in parts[1:]], dtype=torch.float32)
                except ValueError as exc:
                    raise ValueError(f"Invalid vector row at {path}:{line_number}") from exc
                if dimensions is None:
                    dimensions = vector.shape[0]
                elif vector.shape[0] != dimensions:
                    raise ValueError(
                        f"Inconsistent embedding dimension at {path}:{line_number}. "
                        f"Expected {dimensions}, got {vector.shape[0]}."
                    )
                vectors[word] = cls._normalize(vector)
        if dimensions is None:
            raise ValueError(f"No embeddings found in {path}.")
        return cls(vectors, dimensions=dimensions)

    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self._encode_one(text) for text in texts])

    def _encode_one(self, text: str) -> torch.Tensor:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        tokens = [token for token in text.strip().lower().split() if token]
        token_vectors = [self.vectors[token] for token in tokens if token in self.vectors]
        if not token_vectors:
            vector = torch.zeros(self.dimensions, dtype=torch.float32)
        else:
            vector = self._normalize(torch.stack(token_vectors).mean(dim=0))
        self.cache[text] = vector
        return vector

    @staticmethod
    def _normalize(vector: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector)
        if float(norm.item()) == 0.0:
            return vector
        return vector / norm


class SentenceTransformerEmbedder(BaseTextEmbedder):
    def __init__(self, model_name_or_path: str, *, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path, device=device)

    def encode(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).to(dtype=torch.float32, device="cpu")


def build_text_embedder(
    backend: str,
    *,
    word_vector_path: str | None = None,
    sentence_transformer_model: str | None = None,
    device: str | None = None,
) -> BaseTextEmbedder:
    if backend in {"glove", "word2vec"}:
        if not word_vector_path:
            raise ValueError(f"embedding_backend='{backend}' requires --embedding_model_path.")
        return StaticWordVectorEmbedder.from_text_file(word_vector_path)
    if backend == "sentence_embeddings":
        model_name = sentence_transformer_model or "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model_name, device=device)
    raise ValueError(f"Unsupported embedding backend: {backend}")
