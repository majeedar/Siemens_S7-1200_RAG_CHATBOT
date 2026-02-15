#!/usr/bin/env python3
"""One-time ingestion script for the S7-1200 system manual.

Usage:
    python -m scripts.ingest_documents [--pdf PATH] [--recreate]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.ingestion.pdf_processor import S7ManualProcessor
from src.retrieval.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "data" / "ingestion.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest S7-1200 manual into Qdrant")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file")
    parser.add_argument("--recreate", action="store_true", help="Recreate the collection from scratch")
    args = parser.parse_args()

    start = time.time()
    logger.info("=" * 60)
    logger.info("S7-1200 Manual Ingestion Pipeline")
    logger.info("=" * 60)

    # Resolve PDF path
    pdf_path = Path(args.pdf) if args.pdf else settings.pdf_file
    logger.info("PDF path: %s", pdf_path)

    if not pdf_path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        logger.error("Place the S7-1200 manual in data/raw/ or specify --pdf PATH")
        sys.exit(1)

    # Step 1: Process PDF
    logger.info("--- Step 1: Processing PDF ---")
    processor = S7ManualProcessor(pdf_path=pdf_path)
    documents = processor.process()
    logger.info("Created %d document chunks", len(documents))

    # Step 2: Initialize vector store
    logger.info("--- Step 2: Setting up Qdrant ---")
    store = QdrantVectorStore()
    store.create_collection(recreate=args.recreate)

    # Check if collection already has data
    info = store.get_collection_info()
    existing_count = info.get("points_count", 0)
    if existing_count and not args.recreate:
        logger.warning(
            "Collection already contains %d documents. Use --recreate to start fresh.",
            existing_count,
        )
        response = input("Continue and add documents anyway? [y/N]: ").strip().lower()
        if response != "y":
            logger.info("Aborted by user.")
            sys.exit(0)

    # Step 3: Add documents
    logger.info("--- Step 3: Embedding and storing documents ---")
    ids = store.add_documents(documents, batch_size=settings.EMBEDDING_BATCH_SIZE)
    logger.info("Stored %d document vectors", len(ids))

    # Step 4: Verify
    logger.info("--- Step 4: Verification ---")
    final_info = store.get_collection_info()
    logger.info("Collection info: %s", final_info)

    # Run a test query
    test_query = "S7-1200 CPU specifications"
    test_results = store.similarity_search(test_query, k=3)
    logger.info("Test query '%s' returned %d results", test_query, len(test_results))
    for i, doc in enumerate(test_results, 1):
        page = doc.metadata.get("page", "?")
        logger.info("  Result %d: Page %s — %s...", i, page, doc.page_content[:80])

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Ingestion complete in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("Total documents: %d", final_info.get("points_count", len(ids)))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
