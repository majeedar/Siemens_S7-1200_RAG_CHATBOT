#!/usr/bin/env bash
# Automated startup script for S7-1200 RAG Chatbot
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "  S7-1200 RAG Chatbot - Startup Script"
echo "=============================================="

# --- Helper functions ---
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 is not installed."
        return 1
    fi
    echo "  [OK] $1 found"
    return 0
}

wait_for_service() {
    local url="$1"
    local name="$2"
    local max_wait="${3:-60}"
    echo "Waiting for $name at $url..."
    for i in $(seq 1 "$max_wait"); do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "  [OK] $name is ready"
            return 0
        fi
        sleep 1
    done
    echo "  [WARN] $name did not respond within ${max_wait}s"
    return 1
}

# --- Step 1: Check prerequisites ---
echo ""
echo "Step 1: Checking prerequisites..."
check_command docker
check_command docker-compose || check_command "docker compose"

# --- Step 2: Create .env if missing ---
echo ""
echo "Step 2: Checking configuration..."
if [ ! -f .env ]; then
    echo "  Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "  [OK] .env created. Review and update if needed."
    else
        echo "  [WARN] No .env.example found. Using defaults."
    fi
else
    echo "  [OK] .env exists"
fi

# --- Step 3: Check for PDF ---
echo ""
echo "Step 3: Checking for PDF manual..."
PDF_FOUND=false
if ls data/raw/*.pdf 1> /dev/null 2>&1; then
    PDF_FOUND=true
    echo "  [OK] PDF found in data/raw/"
else
    echo "  [WARN] No PDF found in data/raw/"
    echo "  Place the S7-1200 system manual PDF in data/raw/ before ingestion."
fi

# --- Step 4: Start Docker services ---
echo ""
echo "Step 4: Starting Docker services..."
if command -v docker-compose &> /dev/null; then
    docker-compose -f docker/docker-compose.yml up -d qdrant ollama
else
    docker compose -f docker/docker-compose.yml up -d qdrant ollama
fi

# --- Step 5: Wait for services ---
echo ""
echo "Step 5: Waiting for services..."
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

wait_for_service "$QDRANT_URL" "Qdrant" 30
wait_for_service "$OLLAMA_URL" "Ollama" 30

# --- Step 6: Pull Ollama model ---
echo ""
echo "Step 6: Pulling Ollama model..."
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b-instruct-q4_K_M}"
echo "  Model: $OLLAMA_MODEL"
docker exec -it "$(docker ps -q -f ancestor=ollama/ollama)" ollama pull "$OLLAMA_MODEL" 2>/dev/null || \
    curl -s "$OLLAMA_URL/api/pull" -d "{\"name\": \"$OLLAMA_MODEL\"}" > /dev/null || \
    echo "  [WARN] Could not pull model. Pull manually: ollama pull $OLLAMA_MODEL"

# --- Step 7: Install Python dependencies ---
echo ""
echo "Step 7: Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt --quiet
    echo "  [OK] Dependencies installed"
else
    echo "  [WARN] requirements.txt not found"
fi

# --- Step 8: Run ingestion if needed ---
echo ""
echo "Step 8: Checking vector store..."
COLLECTION_EXISTS=$(curl -s "$QDRANT_URL/collections/s7_1200_manual" 2>/dev/null | grep -c '"status":"ok"' || true)

if [ "$COLLECTION_EXISTS" -eq 0 ] && [ "$PDF_FOUND" = true ]; then
    echo "  Collection not found. Running ingestion..."
    python -m scripts.ingest_documents
elif [ "$COLLECTION_EXISTS" -gt 0 ]; then
    echo "  [OK] Collection already exists"
else
    echo "  [SKIP] No PDF available for ingestion"
fi

# --- Step 9: Launch Gradio app ---
echo ""
echo "=============================================="
echo "  Starting Gradio application..."
echo "  Access at: http://localhost:7860"
echo "=============================================="
python -m src.ui.gradio_app
