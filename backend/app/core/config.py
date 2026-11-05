import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "I.See Agents — Multi-Agent AI Platform"
    APP_VERSION: str = "2.0.0"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY", "isee-agents-dev-secret-key-change-in-production-488"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # Database: PostgreSQL + pgvector in Docker/Debian production, SQLite fallback for local standalone execution
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./isee_agents.db")

    # Ollama & Model Providers
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_TIMEOUT_SECONDS: float = 45.0
    ALLOW_DEMO_FALLBACK: bool = True

    # RAG & Upload Security
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./storage/uploads")
    MAX_UPLOAD_SIZE_MB: int = 15
    DEFAULT_CHUNK_SIZE: int = 600
    DEFAULT_CHUNK_OVERLAP: int = 100
    EMBEDDING_DIMENSION: int = 384

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
