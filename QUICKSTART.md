# S7-1200 RAG Chatbot - 5-Minute Quick Start

## Step 1: Start infrastructure (1 min)

```bash
docker compose -f docker/docker-compose.yml up -d qdrant ollama
```

## Step 2: Pull the LLM model (2-3 min, one-time)

```bash
docker exec -it s7-ollama ollama pull llama3.1:8b-instruct-q4_K_M
```

## Step 3: Install Python deps (30 sec)

```bash
pip install -r requirements.txt
```

## Step 4: Add your PDF

Place `s71200_system_manual_en-US_en-US.pdf` into `data/raw/`.

## Step 5: Ingest the manual (10-15 min, one-time)

```bash
python -m scripts.ingest_documents
```

## Step 6: Launch

```bash
python -m src.ui.gradio_app
```

Open **http://localhost:7860** and start asking questions!

## One-liner (Linux/Mac)

```bash
bash scripts/start.sh
```
