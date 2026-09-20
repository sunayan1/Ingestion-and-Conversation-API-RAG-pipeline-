# RAG Pipeline — Document Ingestion & Conversational RAG API

A FastAPI backend implementing two REST APIs:

1. **Document Ingestion API** — upload PDF/TXT files, chunk them using two
   selectable strategies, generate embeddings, and store them in a vector
   database with accompanying metadata.
2. **Conversational RAG API** — a custom-built retrieval-augmented generation
   chat endpoint (no `RetrievalQAChain`, no FAISS/Chroma) with Redis-backed
   multi-turn memory and LLM-based interview booking.

## Tech Stack

| Concern              | Choice                              |
|-----------------------|---------------------------------------|
| API framework          | FastAPI                              |
| Package manager        | `uv`                                  |
| PDF text extraction    | `pypdf`                              |
| Embedding model         | `sentence-transformers/all-mpnet-base-v2` (local, CPU) |
| Vector database         | Qdrant (self-hosted via Docker)      |
| Metadata database        | SQLite via SQLAlchemy ORM            |
| Chat memory              | Redis (self-hosted via Docker)       |
| LLM                        | Groq API (`openai/gpt-oss-20b`)      |

## Why these choices

- **Local embedding model (`all-mpnet-base-v2`) instead of an API-based one**
  (e.g. OpenAI embeddings): no API key or billing required to run this
  project, works fully offline after the first model download, and avoids
  putting a hard external dependency between a grader and a working demo.
  `all-mpnet-base-v2` was chosen over the smaller `all-MiniLM-L6-v2` for
  better retrieval quality, while still running acceptably on CPU.
- **Qdrant over Pinecone/Weaviate/Milvus**: runs locally via a single Docker
  command, no cloud account or API key needed, and has a straightforward
  Python client.
- **SQLite over Postgres/MongoDB**: zero setup — no separate database server
  to install or configure, appropriate for this project's scale.
- **Groq over OpenAI**: free tier with no billing setup required, and very
  fast inference. Rate limits are generous enough for normal testing and
  are not enforced client-side, since exceeding them during light manual
  testing is unlikely.
- **Custom retrieval instead of `RetrievalQAChain`**: retrieval, prompt
  construction, and the LLM call are implemented as separate, explicit
  functions (`retrieval.py`, `rag.py`, `llm.py`) rather than a pre-built
  LangChain chain, per the task constraints.

## Project Structure

```
src/rag_pipeline/
├── main.py                      # creates the FastAPI app, wires up both routers
│
├── ingestion/
│   ├── router.py                 # POST /ingest/
│   ├── extractor.py               # PDF/TXT text extraction
│   ├── chunker.py                  # fixed-size and recursive chunking strategies
│   ├── embeddings.py                # sentence-transformers wrapper
│   └── vector_store.py               # Qdrant collection setup + upsert
│
├── conversation/
│   ├── router.py                 # POST /chat/
│   ├── memory.py                  # Redis-backed chat history per session
│   ├── retrieval.py                # embeds a query, searches Qdrant
│   ├── rag.py                       # builds the prompt from history + retrieved chunks
│   ├── llm.py                        # Groq API wrapper
│   ├── booking.py                     # LLM-based booking intent detection + field extraction
│   └── schemas.py                      # Pydantic request/response models
│
└── db/
    ├── models.py                 # SQLAlchemy models: Documents, Bookings
    └── session.py                  # DB session factory + save functions
```

## Setup

### 1. Prerequisites

- Python 3.14+, [`uv`](https://docs.astral.sh/uv/) installed
- Docker installed and running
- A free [Groq API key](https://console.groq.com/keys)

### 2. Install dependencies

Dependencies are managed via `uv` (`pyproject.toml` + `uv.lock`) rather than
a `requirements.txt`. Install `uv` if you don't have it
(`pip install uv`, or see [the uv docs](https://docs.astral.sh/uv/)), then:

```bash
uv sync
```

This installs every dependency pinned in `uv.lock` into a local `.venv`.

### 3. Start Qdrant and Redis

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
docker run -d -p 6379:6379 redis
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 5. Run the server

```bash
uv run fastapi dev
```

The API is now available at `http://127.0.0.1:8000`, with interactive docs
at `http://127.0.0.1:8000/docs`.

## Key Implementation Details

This section shows the core pieces that satisfy the task's specific
constraints — custom chunking, custom retrieval (no `RetrievalQAChain`),
and manual prompt construction.

### Chunking strategy 1 — fixed-size with overlap

`src/rag_pipeline/ingestion/chunker.py`

```python
def fixed_chunking(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Splits text into fixed-size chunks with overlap between consecutive chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks
```

Cuts text at fixed character intervals, stepping forward by
`chunk_size - chunk_overlap` each time so consecutive chunks share a small
window of context — reducing the chance that a sentence gets split with no
surrounding context in either chunk.

### Chunking strategy 2 — recursive, structure-aware

`src/rag_pipeline/ingestion/chunker.py`

```python
def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    separators: tuple[str, ...] = ("\n\n", ". ", " "),
) -> list[str]:
    """
    Splits text into chunks no larger than chunk_size, preferring natural
    boundaries (paragraphs, then sentences, then words) before falling back
    to a raw fixed-size split. Filters out fragments below a minimum length
    to avoid storing low-information chunks (e.g. standalone headings).
    """
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return fixed_chunking(text, chunk_size)

    current_separator = separators[0]
    remaining_separators = separators[1:]

    pieces = text.split(current_separator)
    result: list[str] = []
    MIN_CHUNK_LENGTH = 30

    for piece in pieces:
        piece = piece.strip()
        if len(piece) < MIN_CHUNK_LENGTH:
            continue
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            result.extend(recursive_chunk(piece, chunk_size, tuple(remaining_separators)))

    return result
```

Tries the most natural split point first (paragraphs), and only recurses
into a finer-grained separator (sentences, then words) for any piece that's
still too large — producing more semantically coherent chunks than a blind
fixed-size cut.

### Custom retrieval — no `RetrievalQAChain`, no FAISS/Chroma

`src/rag_pipeline/conversation/retrieval.py`

```python
from rag_pipeline.ingestion.embeddings import embed
from rag_pipeline.ingestion.vector_store import _client, COLLECTION_NAME


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[str]:
    """Embeds the query and searches Qdrant directly for the most similar stored chunks."""
    query_embedding = embed([query])[0]

    result = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    )

    return [point.payload["text"] for point in result.points]
```

Embeds the incoming query with the same model used at ingestion time, then
queries Qdrant directly for the nearest vectors by cosine similarity — no
LangChain retrieval chain involved.

### Manual RAG prompt construction

`src/rag_pipeline/conversation/rag.py`

```python
from rag_pipeline.conversation.retrieval import retrieve_relevant_chunks
from rag_pipeline.conversation.llm import generate_response


def get_rag_response(session_id: str, question: str, history: list[dict]) -> str:
    """Builds a RAG prompt from retrieved context + chat history, and gets the LLM's answer."""

    relevant_chunks = retrieve_relevant_chunks(question)
    context = "\n\n".join(relevant_chunks)

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant answering questions based on the provided "
            "document context. If the answer isn't in the context, say you don't know "
            "rather than making something up."
        ),
    }

    user_message_with_context = {
        "role": "user",
        "content": f"Context from documents:\n{context}\n\nQuestion: {question}",
    }

    messages = [system_message] + history + [user_message_with_context]

    return generate_response(messages)
```

Retrieved chunks and prior conversation turns are assembled into a plain
message list by hand — the same structure a chain would build internally,
but constructed explicitly here so every step is visible and controllable.

## API Reference

### `POST /ingest/`

Uploads a document, extracts its text, chunks it, embeds the chunks, and
stores them in Qdrant + SQLite.

**Request:** `multipart/form-data`
- `file` (required) — a `.pdf` or `.txt` file
- `strategy` (query param, default `"fixed"`) — `"fixed"` or `"recursive"`

**Response:**
```json
{
  "document_id": "uuid",
  "filename": "example.pdf",
  "chunking_strategy": "fixed",
  "chunk_count": 42
}
```

**Chunking strategies:**
- `fixed` — splits text into fixed-size windows (default 500 characters)
  with a configurable overlap (default 50 characters), to preserve context
  across chunk boundaries.
- `recursive` — attempts to split on natural boundaries first (paragraphs,
  then sentences, then words), falling back to fixed-size splitting only
  when a piece is still too large. Produces more semantically coherent
  chunks at the cost of more variable chunk sizes. Chunks shorter than a
  minimum length (30 characters) are discarded, to avoid storing
  low-information fragments (e.g. standalone headings) that can otherwise
  rank artificially high in similarity search.

### `POST /chat/`

A multi-turn conversational endpoint. Retrieves relevant chunks from Qdrant,
combines them with conversation history from Redis, and generates a
response via Groq — unless the message is detected as an interview booking
request, in which case it extracts and stores the booking instead.

**Request:**
```json
{
  "session_id": "optional — omit or set to null to start a new conversation",
  "message": "your question or message"
}
```

**Response:**
```json
{
  "session_id": "uuid — reuse this in your next request to continue the conversation",
  "answer": "the model's response"
}
```

**Multi-turn behavior:** if `session_id` is omitted, a new one is generated
and returned. Passing that same `session_id` back on the next request
continues the same conversation, with full prior history available to the
model.

**Interview booking:** if a message expresses intent to book an interview
and includes a name, email, date, and time, the booking is detected and
extracted by the LLM, saved to the database, and confirmed in the response
instead of going through the normal RAG path. If any required field is
missing, the message is treated as a normal chat message.

## Known Limitations

- **PDF text extraction is not OCR-aware.** Scanned PDFs or PDFs exported
  from slide decks as flattened images have no embedded text layer and will
  extract to little or no text. This is a fundamental limitation of
  text-layer-based extraction (`pypdf`), not something specific to this
  implementation — OCR was intentionally out of scope for this task.
- **Math-heavy/technical PDFs can extract with degraded fidelity** (broken
  symbols, reordered text) due to how such documents encode formatting
  internally, which can reduce retrieval quality for those documents
  specifically.
- **Retrieval currently searches across the entire Qdrant collection**,
  not scoped to a single document. In practice, cosine similarity still
  favors chunks relevant to the query, but a production version would
  likely support filtering retrieval by `document_id` for multi-document
  workspaces.
- **Embedding runs on CPU**, which is slow for very large documents (multi-
  minute processing for multi-megabyte files). A production deployment
  would use GPU inference or a hosted embedding API for better throughput.
- **Qdrant upserts are batched (100 points per request)** to stay under
  Qdrant's request size limit — necessary for large documents that produce
  many chunks.

## Testing

Interactive testing is available via the Swagger UI at `/docs`. A typical
flow:

1. `POST /ingest/` with a text-based PDF or `.txt` file.
2. `POST /chat/` with a question about that document (omit `session_id`).
3. `POST /chat/` again, reusing the returned `session_id`, with a follow-up
   question to confirm multi-turn memory works.
4. `POST /chat/` with a message like *"I'd like to book an interview, I'm
   Jane Doe, jane@example.com, available Oct 5th at 2pm"* to test booking.
