from sentence_transformers import SentenceTransformer


_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed(text: list[str]) -> list[list[float]]:
    embeddings = _model.encode(text)
    return embeddings.tolist()