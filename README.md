# RAG-Powered Document Assistant — Tech / Office Equipment Manuals

A Retrieval-Augmented Generation (RAG) web app that answers questions about a
collection of equipment manuals, with grounded and cited answers. Includes an
optional Computer Vision (YOLOv8) component that recognizes equipment in an
uploaded photo and feeds that into the RAG context.

## Overview

<!-- 2-3 sentences: what the app does, who it's for, core flow (question -> retrieval -> grounded answer). -->

## Architecture Diagram

<!-- Add an image or ASCII diagram, e.g.:
User -> Streamlit Frontend -> FastAPI Backend -> [Retrieval: Chroma] + [Generation: Ollama LLM] -> Answer + Sources
-->

## Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic
- **Vector Store:** ChromaDB
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM:** Ollama (local, e.g. `llama3`)
- **Frontend:** Streamlit
- **Vision (Extended):** YOLOv8 (Ultralytics, pretrained on COCO)

## Project Structure

```
rag-assistant-app/
├── notebooks/
│   └── rag_pipeline.ipynb
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   ├── services/retrieval.py
│   │   ├── services/generation.py
│   │   └── utils/logging_config.py
│   ├── data/vector_store/
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── .env.example
│   └── requirements.txt
├── README.md
└── .gitignore
```

## Domain & Data

<!-- Describe: what documents (manuals) you collected, how many, where from.
If the raw corpus is excluded from git (see .gitignore), explain here how to
obtain / regenerate it (e.g. a Google Drive link or a script). -->

## Setup — Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # edit values if needed
ollama pull llama3                # make sure Ollama is running locally

uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

Run tests:
```bash
pytest
```

## Setup — Frontend

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

streamlit run app.py
# App: http://localhost:8501
```

## Environment Variables

| Variable | Where | Description | Default |
|---|---|---|---|
| `VECTOR_STORE_DIR` | backend | Path to the persisted Chroma store | `data/vector_store` |
| `EMBEDDING_MODEL_NAME` | backend | sentence-transformers model name | `all-MiniLM-L6-v2` |
| `OLLAMA_MODEL` | backend | Local Ollama model to use | `llama3` |
| `OLLAMA_BASE_URL` | backend | Ollama server URL | `http://localhost:11434` |
| `TOP_K` | backend | Number of chunks retrieved per query | `4` |
| `FRONTEND_ORIGIN` | backend | Allowed CORS origin | `http://localhost:8501` |
| `API_BASE_URL` | frontend | Backend base URL | `http://localhost:8000` |

## API Reference

### `GET /health`
Returns `{"status": "ok"}`

### `POST /query`
Request:
```json
{ "question": "How do I reset the printer?" }
```
Response:
```json
{ "answer": "...", "sources": ["printer_manual.pdf"] }
```

curl example:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset the printer?"}'
```

## Evaluation Results

<!-- Paste the results table from notebook section 2.6, plus a short summary
of failure cases and how they were mitigated. -->

## Screenshots

<!-- Add screenshots of the running Streamlit app here. -->

---

### Verification checklist (before submitting)
- [ ] `notebooks/rag_pipeline.ipynb` runs top-to-bottom (Kernel → Restart & Run All)
- [ ] `backend/`: `/health` + `/query` work, `pytest` passes
- [ ] `frontend/`: chat UI works end-to-end against the backend
- [ ] Cloned into a fresh folder and followed only this README successfully
