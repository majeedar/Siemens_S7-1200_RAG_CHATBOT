"""RAG retrieval quality tracking and citation accuracy checking."""

import logging
import re

from langchain.schema import Document

logger = logging.getLogger(__name__)


class RAGQualityTracker:
    """Tracks retrieval relevance scores and citation accuracy."""

    def __init__(self, low_score_threshold: float = 0.3):
        self.low_score_threshold = low_score_threshold

    def evaluate_retrieval(self, query: str, docs: list[Document]) -> dict:
        """Compute quality metrics for a retrieval result."""
        scores = [doc.metadata.get("score", 0.0) for doc in docs]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        low_quality = len(docs) > 0 and all(
            s < self.low_score_threshold for s in scores
        )

        if low_quality:
            logger.warning(
                "Low quality retrieval: query='%s' avg_score=%.3f num_results=%d",
                query[:80],
                avg_score,
                len(docs),
            )

        return {
            "avg_relevance_score": round(avg_score, 4),
            "min_relevance_score": round(min_score, 4),
            "max_relevance_score": round(max_score, 4),
            "num_results": len(docs),
            "low_quality_retrieval": low_quality,
        }

    def evaluate_citations(
        self, citations: list[str], retrieved_pages: list[int]
    ) -> dict:
        """Check if cited pages match retrieved chunk pages."""
        cited_pages: set[int] = set()
        for cite in citations:
            nums = re.findall(r"\d+", cite)
            cited_pages.update(int(n) for n in nums)

        retrieved_set = set(retrieved_pages)
        matched = cited_pages & retrieved_set
        unmatched = cited_pages - retrieved_set

        return {
            "cited_pages": sorted(cited_pages),
            "matched_pages": sorted(matched),
            "unmatched_pages": sorted(unmatched),
            "citation_accuracy": (
                len(matched) / len(cited_pages) if cited_pages else 1.0
            ),
        }
