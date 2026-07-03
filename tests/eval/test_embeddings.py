from pathlib import Path

import torch

from src.eval.embeddings import StaticWordVectorEmbedder, build_text_embedder


def test_static_word_vector_embedder_loads_headered_text_format(tmp_path: Path):
    embedding_file = tmp_path / "toy.vec"
    embedding_file.write_text(
        "3 2\n"
        "fire 1 0\n"
        "water 0 1\n"
        "steam 1 1\n",
        encoding="utf-8",
    )

    embedder = StaticWordVectorEmbedder.from_text_file(embedding_file)
    encoded = embedder.encode(["fire water"])

    assert encoded.shape == (1, 2)
    assert torch.linalg.vector_norm(encoded[0]).item() > 0.99


def test_build_text_embedder_requires_path_for_static_backends():
    try:
        build_text_embedder("glove")
    except ValueError as exc:
        assert "--embedding_model_path" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing embedding path.")
