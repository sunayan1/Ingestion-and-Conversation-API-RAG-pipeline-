from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "documents"
VECTOR_SIZE = 768

def ensure_collection_exists(): 
    existing_collections = _client.get_collections().collections 
    existing_names = [c.name for c in existing_collections]

    if COLLECTION_NAME not in existing_names: 
        _client.create_collection(
            collection_name = COLLECTION_NAME,
            vectors_config= VectorParams(size = VECTOR_SIZE, distance = Distance.COSINE)
        )

def upsert_chunks(chunks: list[str], vectors: list[list[float]], document_id: str, filename: str, batch_size: int = 100) -> None: 
    points = []
    for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)): 
        point = PointStruct(
            id=str(uuid.uuid4()), 
            vector=vector, 
            payload={
                "text": chunk_text, 
                "document_id": document_id, 
                "chunk_index": i, 
                "filename": filename, 
            },
        )
        points.append(point)
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        _client.upsert(collection_name=COLLECTION_NAME, points=batch)