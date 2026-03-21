import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_project_root, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

load_dotenv(os.path.join(_project_root, ".env"))

from multiagent_rag.utils.logger import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  MULTI-AGENT RAG SYSTEM - Starting up...")
    logger.info("=" * 60)

    try:
        from multiagent_rag.utils.db_client import PineconeClient
        _ = PineconeClient()
        logger.info("Pinecone client initialized")
    except Exception as e:
        logger.error(f"Pinecone initialization failed: {e}")

    try:
        from multiagent_rag.utils.embeddings import EmbeddingManager
        _ = EmbeddingManager()
        logger.info("Dense embedding model loaded")
    except Exception as e:
        logger.error(f"Embedding model loading failed: {e}")

    try:
        from multiagent_rag.utils.sparse import SparseEmbeddingManager
        _ = SparseEmbeddingManager()
        logger.info("Sparse encoder loaded")
    except Exception as e:
        logger.error(f"Sparse encoder loading failed: {e}")

    logger.info("=" * 60)
    logger.info("  System ready. Accepting requests.")
    logger.info("=" * 60)

    yield

    logger.info("Multi-Agent RAG System shutting down.")


app = FastAPI(title="Multi-Agent RAG System API", version="1.0.0", lifespan=lifespan, )

app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000",
                                  "http://localhost:8501", "*", ], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"], )

from api.routes.chat import router as chat_router
from api.routes.ingestion import router as ingestion_router
from api.routes.knowledge import router as knowledge_router
from api.routes.health import router as health_router
from api.routes.feedback import router as feedback_router
from api.routes.tools_router import router as tools_router
from api.handoff import router as handoff_router
from api.session_api import router as session_router

app.include_router(chat_router)
app.include_router(ingestion_router)
app.include_router(knowledge_router)
app.include_router(health_router)
app.include_router(feedback_router)
app.include_router(tools_router)
app.include_router(handoff_router)
app.include_router(session_router)


@app.get("/")
async def root():
    return {"message": "Multi-Agent RAG System API", "docs": "/docs", "health": "/api/health", }
