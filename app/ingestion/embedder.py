import numpy as np
from typing import List, Union
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

settings = get_settings()


class Embedder:
    def __init__(self):
        print(f"[Embedder] Loading model '{settings.embedding_model}'...")
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Ready. Vector dimension: {self.dimension}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        ).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()