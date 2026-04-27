import numpy as np
from typing import List, Union
from functools import lru_cache
from app.core.config import get_settings

settings = get_settings()


class Embedder:
    """
    Uses FastEmbed instead of sentence-transformers.
    No PyTorch. No GPU needed. Runs fine on 512MB RAM.
    Same 384-dimensional vectors, same quality.
    """

    def __init__(self):
        print("[Embedder] Loading FastEmbed model...")
        from fastembed import TextEmbedding
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.dimension = 384
        print(f"[Embedder] Ready. Dimension: {self.dimension}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = list(self.model.embed(texts))
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()