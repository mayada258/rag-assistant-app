"""
Generation service.

Builds a grounded prompt from retrieved chunks and calls the local Ollama LLM.
"""

import ollama

from app.core.config import settings
PROMPT_TEMPLATE = """You are a helpful assistant. Use the context below to answer
the question as best you can, even if the context only partially covers it.
Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(
    question: str, chunks: list[dict], detected_objects: list[str] | None = None
) -> str:
    if not chunks:
        return "I don't have enough information in the documents to answer that."

    if detected_objects:
        question = f"[Image shows: {', '.join(detected_objects)}] {question}"

    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=settings.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]
