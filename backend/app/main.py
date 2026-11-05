from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import settings
from app.db.session import Base, SessionLocal, engine, ensure_pgvector_extension
from app.services.agent_registry import seed_initial_platform_data


def initialize_platform_database() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        ensure_pgvector_extension(db)
        seed_initial_platform_data(db)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_platform_database()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Multi-Agent AI Platform for agent management, orchestration, RAG knowledge bases, "
        "LoRA/QLoRA specialization, and Ollama local inference — I.See Technologies."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def healthcheck():
    return {
        "status": "ok",
        "platform": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "developer": "Júlio César — I.See Technologies",
    }
