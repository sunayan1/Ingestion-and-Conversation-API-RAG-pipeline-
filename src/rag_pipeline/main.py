from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline.ingestion.router import router as ingestion_router
from rag_pipeline.ingestion.vector_store import ensure_collection_exists
from rag_pipeline.conversation.router import router as conversation_router

app = FastAPI()

ensure_collection_exists()  # runs once when the app starts

app.include_router(ingestion_router, prefix="/ingest", tags=["ingestion"])
app.include_router(conversation_router, prefix="/chat", tags=["conversation"])

