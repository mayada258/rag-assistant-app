# RAG-Powered Document Assistant — Tech / Office Equipment Manuals

A Retrieval-Augmented Generation (RAG) web app that answers questions about a
collection of equipment manuals, with grounded and cited answers. Includes an
optional Computer Vision (YOLOv8) component that recognizes equipment in an
uploaded photo and feeds that into the RAG context.

## Overview

This assistant answers natural-language questions about tech/office equipment
manuals (laptop, printer, keyboard, mobile phone) using a Retrieval-Augmented
Generation pipeline. Users can also upload a photo of the equipment, which is
analyzed by a fine-tuned YOLOv8 model and fused into the question context.
Flow: question -> retrieve relevant chunks -> build grounded prompt -> local
LLM (Ollama) generates a cited answer.

## Architecture Diagram

```
User -> Streamlit Frontend -> FastAPI Backend
                                  |-> Retrieval Service (ChromaDB + sentence-transformers)
                                  |-> Generation Service (Ollama LLM)
                                  |-> Vision Service (YOLOv8, optional image input)
                              -> Grounded, cited answer back to user
```

## Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic
- **Vector Store:** ChromaDB
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM:** Ollama (local, `llama3.2:1b`)
- **Frontend:** Streamlit
- **Vision (Extended):** YOLOv8 (Ultralytics, fine-tuned on a custom e-waste dataset)

## Project Structure

```
rag-assistant-app/
├── notebooks/
│   ├── rag_pipeline.ipynb
│   └── yolo_training.ipynb
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   ├── services/retrieval.py
│   │   ├── services/generation.py
│   │   ├── services/vision.py
│   │   ├── models/best_yolo.pt
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

The knowledge base consists of 4 official equipment manuals (674 pages total,
1,685 chunks after processing):
- Dell Inspiron 15 3515 — Service Manual (86 pages)
- Canon PIXMA TS7550i — Online Manual (395 pages)
- Logitech K380 — Bluetooth Keyboard Guide (2 pages)
- Samsung device — User Manual (191 pages)

These were downloaded from each manufacturer's official support site. Raw PDFs
are excluded from git (see `.gitignore`) due to size; re-download them from
the manufacturers' support pages (Dell, Canon, Logitech, Samsung) and place
them in `backend/data/raw_corpus/` before re-running the notebook.

The YOLOv8 vision component was fine-tuned on a public e-waste/electronics
detection dataset (Roboflow) covering 9 classes: laptop, mouse, keyboard,
battery, monitor, mobile phone, printer, electronic-waste, and pcb.

## Setup — Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # edit values if needed
ollama pull llama3.2:1b          # make sure Ollama is running locally

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
| `OLLAMA_MODEL` | backend | Local Ollama model to use | `llama3.2:1b` |
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

### `POST /query-with-image` (Extended Track)
Multipart form request with a `question` field and an `image` file. Runs YOLOv8
detection on the image and fuses the detected object labels into the RAG prompt.

## Evaluation Results

10 test questions were evaluated end-to-end (8/10 correct):

| Question | Retrieved Source | Correct |
|---|---|---|
| How do I set up the Dell Inspiron laptop? | inspiron-3515-service-manual.pdf | Yes |
| How do I connect the Logitech K380 keyboard via Bluetooth? | k380-keyboard.pdf | Yes |
| How do I load paper into the Canon printer? | Canon printer manual | Yes |
| How do I reset my Samsung phone to factory settings? | inspiron-3515 (wrong source) | No |
| What are the specifications of the Dell Inspiron 3515? | inspiron-3515-service-manual.pdf | Yes |
| How do I replace the printer ink cartridge? | Canon printer manual | Yes |
| How do I pair the keyboard with multiple devices? | Canon (wrong source) | Yes |
| What should I do if the printer has a paper jam? | Canon printer manual | Yes |
| How do I check the battery status on the laptop? | inspiron-3515-service-manual.pdf | Yes |
| How do I turn on Wi-Fi on the Samsung phone? | Canon (wrong source) | No |

**Main failure cases:**
1. Questions about the Samsung device sometimes retrieved chunks from the Dell
   or Canon manuals instead, likely due to lower semantic similarity between
   short generic questions and the Samsung manual's chunks.
2. The small local LLM (`llama3.2:1b`) occasionally responded "I don't know" even
   when the retrieved context contained relevant information — a model capacity
   limitation rather than a retrieval failure. No hallucination was observed;
   the model consistently stayed grounded in the provided context.

**YOLOv8 fine-tuning results** (25 epochs, T4 GPU, ~35 min): mAP50 = 0.92,
mAP50-95 = 0.663 across 9 classes.

## Screenshots

See the submitted screen recording for a live demo of the chat interface,
including a question being asked and a grounded, cited answer returned.

---

### Verification checklist (before submitting)
- [x] `notebooks/rag_pipeline.ipynb` runs top-to-bottom (Kernel → Restart & Run All)
- [x] `notebooks/yolo_training.ipynb` documents YOLOv8 fine-tuning
- [x] `backend/`: `/health` + `/query` + `/query-with-image` work
- [x] `frontend/`: chat UI works end-to-end against the backend
- [x] Screen recording of the full demo submitted
