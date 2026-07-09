"""Registro de modelos de embeddings elegibles con --model (sentence-transformers o GloVe)."""
from . import config

# key -> (kind, model_name, prefix)
#   'st'    = sentence-transformer contextual
#   'glove' = estático word-level (GloVe 6B.300d)
MODELS = {
    "bge-base": ("st", "BAAI/bge-base-en-v1.5", ""),        # default (contextual)
    "e5-base": ("st", "intfloat/e5-base-v2", "query: "),    # e5 requiere prefijo
    "bge-large": ("st", "BAAI/bge-large-en-v1.5", ""),      # ~1.3GB de descarga
    "gte-base": ("st", "thenlper/gte-base", ""),
    "minilm": ("st", "sentence-transformers/all-MiniLM-L6-v2", ""),  # chico/rápido, 384-dim
    "glove": ("glove", None, ""),                            # estático (GloVe 6B.300d)
}


def build_embedder(model_key: str, use_fake: bool = False):
    """Devuelve (embedder, prefix). El embedder expone `.encode(texts, prefix=...)`."""
    if use_fake:
        from .embed_utils import FakeEmbedder
        return FakeEmbedder(), ""
    kind, name, prefix = MODELS[model_key]
    if kind == "glove":
        from .glove_embed import GloveEmbedder
        return GloveEmbedder(), ""
    from .embed_utils import Embedder
    # bge-base reusa el cache flat existente; el resto va a un subdir por modelo.
    cache_dir = config.CACHE_DIR if name == config.EMBEDDING_MODEL_NAME else f"{config.CACHE_DIR}/{model_key}"
    return Embedder(model_name=name, cache_dir=cache_dir), prefix
