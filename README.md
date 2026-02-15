# S7-1200 PLC RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot that answers technical questions about the Siemens S7-1200 Programmable Logic Controller using the official 864-page system manual.

## Architecture

```
User Question
    |
    v
[Gradio UI] --> [RAG Chain] --> [Hybrid Retrieval] --> [Qdrant Vector DB]
                    |                                        |
                    v                                        |
              [Ollama LLM] <-- formatted context <-----------+
                    |
                    v
           Cited Response + Sources
```

**Stack:** Gradio | LangChain | Qdrant | Ollama (Llama 3.1 8B) | sentence-transformers/all-MiniLM-L6-v2

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- ~8 GB disk space (model + vectors)
- GPU recommended (6-8 GB VRAM) or CPU-only with slower inference

### 1. Clone and configure

```bash
cd siemens_S7-1200_RAG
cp .env.example .env
```

### 2. Add the PDF manual

Place `s71200_system_manual_en-US_en-US.pdf` into `data/raw/`.

### 3. Start services

```bash
# Start Qdrant and Ollama
docker compose -f docker/docker-compose.yml up -d qdrant ollama

# Pull the LLM model
docker exec -it s7-ollama ollama pull llama3.1:8b-instruct-q4_K_M

# Install Python dependencies
pip install -r requirements.txt
```

### 4. Ingest the manual

```bash
python -m scripts.ingest_documents
```

This parses all 864 pages, chunks them, generates embeddings, and stores them in Qdrant. Takes ~10-15 minutes.

### 5. Launch the chatbot

```bash
python -m src.ui.gradio_app
```

Open http://localhost:7860 in your browser.

### Alternative: Full Docker deployment

```bash
docker compose -f docker/docker-compose.yml up -d
```

## Project Structure

```
siemens_S7-1200_RAG/
├── data/
│   ├── raw/                    # Place PDF here
│   ├── processed/              # Optional extracted data
│   └── vectorstore/            # Qdrant persistence
├── src/
│   ├── config.py               # Pydantic settings
│   ├── ingestion/
│   │   └── pdf_processor.py    # PDF parsing & chunking
│   ├── retrieval/
│   │   └── vector_store.py     # Qdrant operations
│   ├── chains/
│   │   └── rag_chain.py        # LangChain RAG pipeline
│   └── ui/
│       └── gradio_app.py       # Gradio interface
├── scripts/
│   ├── ingest_documents.py     # One-time ingestion
│   └── start.sh                # Automated startup
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

All settings are managed via environment variables (`.env` file). Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Ollama model name |
| `OLLAMA_TEMPERATURE` | `0.1` | LLM temperature (lower = more precise) |
| `CHUNK_SIZE` | `800` | Document chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `ENABLE_HYBRID_SEARCH` | `true` | Combine vector + keyword search |

See [.env.example](.env.example) for all options.

## Features

- **Hybrid retrieval**: Dense vector similarity + keyword matching for better recall
- **Smart chunking**: Preserves code blocks, tables, and safety warnings
- **Metadata enrichment**: Page numbers, chapters, topic tags, code/table detection
- **Citations**: Every response includes `[Page X]` references
- **Safety awareness**: Automatic highlighting of WARNING/CAUTION/DANGER content
- **Conversation memory**: Context-aware follow-up questions (last 10 exchanges)
- **Adjustable settings**: Temperature and top-K controls in the UI
- **Source transparency**: Full source display panel with scores and metadata

## Example Queries

- "How do I configure analog input AI_0?"
- "What is the maximum I/O capacity of S7-1200?"
- "Explain PID control function blocks"
- "How to connect HMI to S7-1200 CPU?"
- "What are the safety precautions for wiring?"
- "Show me example code for motion control"

## Troubleshooting

**Qdrant not connecting:**
```bash
docker ps  # Check if container is running
curl http://localhost:6333/healthz  # Test health endpoint
```

**Ollama not responding:**
```bash
docker exec -it s7-ollama ollama list  # Check models
curl http://localhost:11434/api/tags   # Test API
```

**Slow responses:**
- Use GPU for Ollama (configure in docker-compose.yml)
- Reduce `RETRIEVAL_TOP_K` to 3
- Use a smaller quantization: `llama3.1:8b-instruct-q4_0`

**Empty or poor results:**
- Re-run ingestion: `python -m scripts.ingest_documents --recreate`
- Verify PDF is in `data/raw/`
- Check Qdrant collection: `curl http://localhost:6333/collections/s7_1200_manual`

## Performance

| Metric | Target |
|--------|--------|
| Ingestion (864 pages) | 10-15 min |
| Query end-to-end | 2-4 sec |
| Retrieval only | <200 ms |
| Memory (RAM) | <12 GB |
| VRAM (GPU) | 6-8 GB |
