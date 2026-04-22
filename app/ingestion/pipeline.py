from typing import Optional
from app.ingestion.chunker import MeetingChunker
from app.ingestion.vector_store import VectorStore


class IngestionPipeline:

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.chunker = MeetingChunker()
        self.vector_store = VectorStore(workspace_id)

    def ingest(
        self,
        source_id: str,
        text: str,
        source_type: str = "transcript",
        metadata: Optional[dict] = None,
    ) -> int:
        meta = {
            "source_type": source_type,
            "workspace_id": self.workspace_id,
            **(metadata or {}),
        }
        print(f"[Pipeline] Ingesting '{source_id}'...")
        chunks = self.chunker.chunk(text, source_id=source_id, metadata=meta)
        print(f"[Pipeline] {len(chunks)} chunks created")
        count = self.vector_store.add(chunks)
        print(f"[Pipeline] Done. Total in store: {self.vector_store.count}")
        return count

    def search(self, query: str, top_k: int = None):
        return self.vector_store.search(query, top_k=top_k)

    def remove(self, source_id: str) -> None:
        self.vector_store.delete_source(source_id)

    @property
    def stats(self) -> dict:
        return {
            "workspace_id": self.workspace_id,
            "total_chunks": self.vector_store.count,
        }