from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    # LLM
    anthropic_api_key: str
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 1024

    # Chroma vector store
    chroma_persist_dir: str = "./chroma_db"

    # RAG settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    min_similarity_score: float = 0.25
    embedding_model: str = "all-MiniLM-L6-v2"

    # App
    app_env: str = "development"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()