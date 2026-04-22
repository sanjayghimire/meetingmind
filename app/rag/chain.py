from anthropic import Anthropic
from app.ingestion.vector_store import VectorStore, SearchResult
from app.core.config import get_settings
from typing import Generator

settings = get_settings()

SYSTEM_PROMPT = """You are MeetingMind, an AI assistant that answers
questions about a team's meetings, decisions, and discussions.

Rules you MUST follow:
1. Answer ONLY using the provided context. Never use outside knowledge.
2. Always cite sources using [Source: source_id] after each claim.
3. If the context does not contain the answer say:
   "I could not find this in your meeting history."
4. Be concise and direct. Lead with the answer.
5. When quoting someone use their exact words and name them."""


def _format_context(results: list[SearchResult]) -> str:
    if not results:
        return "No relevant context found."
    parts = []
    for i, r in enumerate(results, 1):
        source_type = r.metadata.get("source_type", "document")
        speakers = r.metadata.get("speakers", "")
        parts.append(
            f"[Source {i}: {r.source_id} | "
            f"Type: {source_type} | "
            f"Speakers: {speakers} | "
            f"Score: {r.score}]\n{r.chunk_text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_user_message(question: str, context: str) -> str:
    return f"""Here is context retrieved from your meeting history:

========== CONTEXT START ==========
{context}
========== CONTEXT END ==========

Question: {question}

Answer based ONLY on the context above.
Cite every claim with [Source: source_id]."""


class RAGChain:

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.vector_store = VectorStore(workspace_id)
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def query(self, question: str, top_k: int = None) -> dict:
        results = self.vector_store.search(question, top_k=top_k)
        context = _format_context(results)

        response = self.client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": _build_user_message(question, context),
            }],
        )

        return {
            "answer": response.content[0].text,
            "sources": results,
        }

    def query_stream(self, question: str,
                     top_k: int = None) -> Generator[str, None, None]:
        results = self.vector_store.search(question, top_k=top_k)
        context = _format_context(results)

        yield f"__SOURCES__{[r.source_id for r in results]}__SOURCES__"

        with self.client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": _build_user_message(question, context),
            }],
        ) as stream:
            for token in stream.text_stream:
                yield token