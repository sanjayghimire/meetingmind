from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional


app = FastAPI(
    title="MeetingMind API",
    description="RAG-powered meeting intelligence",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    workspace_id: str
    source_id: str
    text: str
    source_type: str = "transcript"
    title: str = "Untitled"


class QueryRequest(BaseModel):
    workspace_id: str
    question: str
    top_k: int = 5


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "MeetingMind"}


@app.post("/api/v1/ingest")
def ingest(req: IngestRequest):
    try:
        from app.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline(workspace_id=req.workspace_id)
        count = pipeline.ingest(
            source_id=req.source_id,
            text=req.text,
            source_type=req.source_type,
            metadata={"title": req.title},
        )
        return {
            "status": "success",
            "source_id": req.source_id,
            "chunks_created": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query")
def query(req: QueryRequest):
    try:
        from app.rag.chain import RAGChain
        chain = RAGChain(workspace_id=req.workspace_id)
        result = chain.query(req.question, top_k=req.top_k)
        return {
            "answer": result["answer"],
            "sources": [
                {
                    "source_id": s.source_id,
                    "score": s.score,
                    "excerpt": s.chunk_text[:200],
                }
                for s in result["sources"]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats/{workspace_id}")
def stats(workspace_id: str):
    from app.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline(workspace_id=workspace_id)
    return pipeline.stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)