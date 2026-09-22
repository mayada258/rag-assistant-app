from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import query
from app.core.config import settings
from app.services.retrieval import load_vector_store
from app.services.vision import load_yolo_model
from app.utils.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    load_vector_store()  # load once at startup, not per request
    load_yolo_model()
    yield


app = FastAPI(title="RAG Document Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
