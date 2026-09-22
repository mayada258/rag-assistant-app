import shutil

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.query import QueryRequest, QueryResponse
from app.services import generation, retrieval
from app.services.vision import detect_objects

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


@router.post("/query-with-image")
async def query_with_image(question: str = Form(...), image: UploadFile = File(...)):
    """Extended Track: accepts a question + an image, runs YOLO detection,
    and fuses the detected objects into the RAG prompt context."""
    temp_path = f"/tmp/{image.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    detected = detect_objects(temp_path)
    chunks = retrieval.retrieve(question)
    answer = generation.generate_answer(question, chunks, detected_objects=detected)
    sources = sorted({c["source"] for c in chunks})

    return {"answer": answer, "sources": sources, "detected_objects": detected}
