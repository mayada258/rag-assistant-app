from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables (.env)."""

    # Vector store
    VECTOR_STORE_DIR: str = "data/vector_store"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # LLM (Ollama)
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Retrieval
    TOP_K: int = 4

    # CORS
    FRONTEND_ORIGIN: str = "http://localhost:8501"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
