"""
Retrieval service.

Loads the persisted Chroma vector store (built in notebooks/rag_pipeline.ipynb)
and exposes a function to retrieve the most relevant chunks for a question.

TODO (fill in once your notebook's vector store is exported):
- Point VECTOR_STORE_DIR (in core/config.py) to the exported folder.
- Make sure the embedding model used here matches the one used to build the store.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import settings

_client = None
_collection = None
_embedder = None


def load_vector_store():
    """Called once at FastAPI startup (see main.py lifespan)."""
    global _client, _collection, _embedder

    _client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
    _collection = _client.get_or_create_collection("rag_chunks")
    _embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """
    Returns a list of dicts: [{"text": ..., "source": ..., "score": ...}, ...]
    """
    if _collection is None or _embedder is None:
        raise RuntimeError("Vector store not loaded. Call load_vector_store() at startup.")

    k = top_k or settings.TOP_K
    query_embedding = _embedder.encode([question]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown") if meta else "unknown",
                "score": dist,
            }
        )
    return chunks
