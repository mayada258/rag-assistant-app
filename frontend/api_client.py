import os

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def ask_question(question: str) -> dict:
    """
    Calls the backend /query endpoint.
    Returns {"answer": str, "sources": list[str]} or raises requests.HTTPError.
    """
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def check_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False
