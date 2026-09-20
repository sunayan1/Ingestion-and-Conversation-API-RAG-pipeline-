from rag_pipeline.ingestion.embeddings import embed
from rag_pipeline.ingestion.vector_store import _client, COLLECTION_NAME

def retrieve_relevant_chunks(query: str, top_k: int = 10): 
    query_embedding = embed([query])[0]
    result = _client.query_points(
        collection_name=COLLECTION_NAME, 
        query=query_embedding,
        limit = top_k
    )
    chunks = [point.payload["text"] for point in result.points] 
    return chunks


