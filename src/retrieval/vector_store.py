"""Qdrant vector store integration for S7-1200 document retrieval.

Handles collection management, document embedding, and hybrid retrieval
(dense vector similarity + sparse BM25 keyword matching).
"""

import logging
import math
from typing import Optional

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Manages the Qdrant collection for S7-1200 manual chunks."""

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.url = url or settings.QDRANT_URL
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.client = QdrantClient(url=self.url, timeout=60)

        logger.info("Loading embedding model: %s", embedding_model or settings.EMBEDDING_MODEL)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model or settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": settings.EMBEDDING_BATCH_SIZE},
        )
        self.vector_size = settings.QDRANT_VECTOR_SIZE

    def collection_exists(self) -> bool:
        """Check whether the collection already exists in Qdrant."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except (UnexpectedResponse, Exception):
            return False

    def create_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection with proper configuration.

        Args:
            recreate: If True, delete and recreate the collection.
        """
        if recreate and self.collection_exists():
            logger.warning("Deleting existing collection: %s", self.collection_name)
            self.client.delete_collection(self.collection_name)

        if self.collection_exists():
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        logger.info("Creating collection '%s'", self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size,
                distance=qmodels.Distance.COSINE,
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            ),
            optimizers_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )

        # Create payload indexes for filtering
        for field, schema in [
            ("metadata.page", qmodels.PayloadSchemaType.INTEGER),
            ("metadata.has_code", qmodels.PayloadSchemaType.BOOL),
            ("metadata.has_safety_warning", qmodels.PayloadSchemaType.BOOL),
            ("metadata.chapter", qmodels.PayloadSchemaType.KEYWORD),
        ]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema,
            )

        logger.info("Collection and indexes created successfully")

    def add_documents(self, documents: list[Document], batch_size: int = 32) -> list[str]:
        """Embed and upsert documents into Qdrant in batches.

        Args:
            documents: LangChain Document objects to store.
            batch_size: Number of documents per batch.

        Returns:
            List of point IDs.
        """
        total = len(documents)
        num_batches = math.ceil(total / batch_size)
        all_ids: list[str] = []

        logger.info("Adding %d documents in %d batches", total, num_batches)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_docs = documents[start:end]

            texts = [doc.page_content for doc in batch_docs]
            metadatas = [doc.metadata for doc in batch_docs]

            # Generate embeddings
            vectors = self.embeddings.embed_documents(texts)

            # Build Qdrant points
            points = []
            for i, (text, meta, vector) in enumerate(zip(texts, metadatas, vectors)):
                point_id = start + i
                payload = {
                    "page_content": text,
                    "metadata": meta,
                }
                points.append(
                    qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
                )
                all_ids.append(str(point_id))

            self.client.upsert(collection_name=self.collection_name, points=points)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                logger.info("Batch %d/%d complete (%d docs)", batch_idx + 1, num_batches, end)

        logger.info("All %d documents added successfully", total)
        return all_ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: Optional[qmodels.Filter] = None,
    ) -> list[Document]:
        """Search for documents similar to the query.

        Args:
            query: The search query text.
            k: Number of results to return.
            score_threshold: Minimum similarity score.
            filter_conditions: Optional Qdrant filter.

        Returns:
            List of matching Document objects.
        """
        query_vector = self.embeddings.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            query_filter=filter_conditions,
            with_payload=True,
        )

        documents: list[Document] = []
        for point in results.points:
            payload = point.payload or {}
            doc = Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {}),
            )
            doc.metadata["score"] = point.score
            documents.append(doc)

        return documents

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Search and return documents with their similarity scores."""
        query_vector = self.embeddings.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )

        docs_and_scores: list[tuple[Document, float]] = []
        for point in results.points:
            payload = point.payload or {}
            doc = Document(
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {}),
            )
            docs_and_scores.append((doc, point.score))

        return docs_and_scores

    def keyword_search(self, query: str, k: int = 5) -> list[Document]:
        """Simple keyword-based search using Qdrant scroll with text matching.

        This supplements the dense vector search for hybrid retrieval.
        """
        # Use scroll with a text match filter for keyword-based retrieval
        keywords = query.lower().split()
        # Search for documents containing key terms
        results: list[Document] = []
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=k * 3,  # Over-fetch then rank
                with_payload=True,
            )
            points = scroll_result[0]

            # Score by keyword overlap
            scored: list[tuple[Document, int]] = []
            for point in points:
                payload = point.payload or {}
                content = payload.get("page_content", "").lower()
                score = sum(1 for kw in keywords if kw in content)
                if score > 0:
                    doc = Document(
                        page_content=payload.get("page_content", ""),
                        metadata=payload.get("metadata", {}),
                    )
                    scored.append((doc, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = [doc for doc, _ in scored[:k]]
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)

        return results

    def hybrid_search(self, query: str, k: int = 5) -> list[Document]:
        """Combine dense vector search with keyword search for better recall.

        Returns deduplicated, merged results.
        """
        # Dense retrieval
        dense_results = self.similarity_search(query, k=k)

        if not settings.ENABLE_HYBRID_SEARCH:
            return dense_results

        # Keyword retrieval
        keyword_results = self.keyword_search(query, k=k)

        # Merge and deduplicate
        seen_pages: set[tuple] = set()
        merged: list[Document] = []

        for doc in dense_results:
            key = (doc.metadata.get("page", 0), doc.page_content[:100])
            if key not in seen_pages:
                seen_pages.add(key)
                merged.append(doc)

        for doc in keyword_results:
            key = (doc.metadata.get("page", 0), doc.page_content[:100])
            if key not in seen_pages:
                seen_pages.add(key)
                merged.append(doc)

        return merged[:k]

    def get_collection_info(self) -> dict:
        """Get information about the current collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
                "vector_size": self.vector_size,
            }
        except Exception as e:
            return {"name": self.collection_name, "error": str(e), "status": "unavailable"}

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.info("Collection '%s' deleted", self.collection_name)
