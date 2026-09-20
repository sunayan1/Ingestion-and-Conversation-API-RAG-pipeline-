import uuid
from fastapi import APIRouter, UploadFile, HTTPException
from pydantic import BaseModel

from rag_pipeline.ingestion.extractor import extract_text
from rag_pipeline.ingestion.chunker import fixed_chunking, recursive_chunk
from rag_pipeline.ingestion.embeddings import embed
from rag_pipeline.ingestion.vector_store import upsert_chunks
from rag_pipeline.db.session import save_document_metadata

router = APIRouter()

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunking_strategy: str
    chunk_count: int

@router.post("/", response_model=IngestResponse)
async def ingest_document(file:UploadFile, strategy: str="fixed") -> IngestResponse: 
    if strategy not in("fixed", "recursive"): 
        raise HTTPException(status_code=400, detail=f"Unknown stategy: {strategy}")

    contents = await file.read()

    try: 
        text = extract_text(file.filename, contents)
    except ValueError as e: 
        raise HTTPException(status_code=400, detail=str(e))

    if strategy == "fixed": 
        chunks = fixed_chunking(text)
    else: 
        chunks = recursive_chunk(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from this file")

    vectors = embed(chunks)

    document_id = str(uuid.uuid4())

    upsert_chunks(chunks=chunks, vectors=vectors, document_id=document_id, filename=file.filename)

    save_document_metadata(
        document_id=document_id, 
        filename=file.filename, 
        chunking_strategy=strategy, 
        chunk_count=len(chunks)
    )

    return IngestResponse(
        document_id=document_id,
        filename=file.filename,
        chunking_strategy=strategy,
        chunk_count=len(chunks)
    )
