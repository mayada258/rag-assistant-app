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


def ask_question_with_image(question: str, image_file) -> dict:
    """
    Calls the backend /query-with-image endpoint.
    image_file: a file-like object (e.g. from st.file_uploader)
    """
    files = {"image": (image_file.name, image_file.getvalue())}
    data = {"question": question}
    response = requests.post(
        f"{API_BASE_URL}/query-with-image",
        data=data,
        files=files,
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
