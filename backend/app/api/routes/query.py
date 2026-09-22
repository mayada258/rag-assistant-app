from fastapi import APIRouter

from app.schemas.query import QueryRequest, QueryResponse
from app.services import generation, retrieval

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    chunks = retrieval.retrieve(request.question)
    answer = generation.generate_answer(request.question, chunks)
    sources = sorted({c["source"] for c in chunks})
    return QueryResponse(answer=answer, sources=sources)
