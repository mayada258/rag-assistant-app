"""
Generation service.

Builds a grounded prompt from retrieved chunks and calls the local Ollama LLM.
"""

import ollama

from app.core.config import settings

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the
context below. If the answer is not contained in the context, say you don't know.
Cite the source of each fact you use in square brackets, e.g. [source: filename.pdf].

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have enough information in the documents to answer that."

    prompt = build_prompt(question, chunks)

    response = ollama.chat(
        model=settings.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]
