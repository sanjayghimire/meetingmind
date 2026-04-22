import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List
from dataclasses import dataclass
from app.ingestion.chunker import Chunk
from app.ingestion.embedder import get_embedder
from app.core.config import get_settings

settings = get_settings()


@dataclass
class SearchResult:
    chunk_text: str
    source_id: str
    score: float
    metadata: dict


def _get_chroma_client():
    import logging
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

def _collection_name(workspace_id: str) -> str:
    return f"ws-{workspace_id[:50]}"


class VectorStore:

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.embedder = get_embedder()
        self._client = _get_chroma_client()
        self._collection = self._client.get_or_create_collection(
            name=_collection_name(workspace_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts).tolist()
        ids = [f"{c.source_id}__chunk_{c.chunk_index}" for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {}
            for k, v in c.metadata.items():
                if isinstance(v, list):
                    meta[k] = ", ".join(str(i) for i in v)
                elif isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            meta["source_id"] = c.source_id
            meta["chunk_index"] = c.chunk_index
            metadatas.append(meta)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[VectorStore] Stored {len(chunks)} chunks")
        return len(chunks)

    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        k = top_k or settings.retrieval_top_k
        query_embedding = self.embedder.embed_one(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        for doc, meta, dist in zip(docs, metas, distances):
            score = 1 - dist
            if score >= settings.min_similarity_score:
                output.append(SearchResult(
                    chunk_text=doc,
                    source_id=meta.get("source_id", "unknown"),
                    score=round(score, 3),
                    metadata=meta,
                ))
        return sorted(output, key=lambda r: r.score, reverse=True)

    @property
    def count(self) -> int:
        return self._collection.count()