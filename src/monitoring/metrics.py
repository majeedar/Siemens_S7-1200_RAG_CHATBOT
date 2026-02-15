"""Prometheus metrics for the S7-1200 RAG chatbot."""

import logging

from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = logging.getLogger(__name__)

# --- Request metrics ---
REQUEST_COUNT = Counter(
    "s7rag_requests_total",
    "Total number of RAG queries",
    ["status"],
)

RESPONSE_TIME = Histogram(
    "s7rag_response_duration_seconds",
    "End-to-end response time",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

RETRIEVAL_TIME = Histogram(
    "s7rag_retrieval_duration_seconds",
    "Vector retrieval time",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

GENERATION_TIME = Histogram(
    "s7rag_generation_duration_seconds",
    "LLM generation time",
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

# --- Quality metrics ---
RETRIEVAL_SCORE = Histogram(
    "s7rag_retrieval_relevance_score",
    "Average relevance score per query",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

LOW_QUALITY_RETRIEVALS = Counter(
    "s7rag_low_quality_retrievals_total",
    "Queries where all results scored below threshold",
)

CITATION_ACCURACY = Histogram(
    "s7rag_citation_accuracy",
    "Fraction of cited pages matching retrieved pages",
    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
)

# --- LLM metrics ---
LLM_ERRORS = Counter(
    "s7rag_llm_errors_total",
    "LLM generation errors by type",
    ["error_type"],
)

TOKENS_PER_SECOND = Histogram(
    "s7rag_tokens_per_second",
    "Approximate tokens/sec during generation",
    buckets=[1, 5, 10, 20, 50, 100],
)

# --- Feedback metrics ---
FEEDBACK_COUNT = Counter(
    "s7rag_feedback_total",
    "User feedback events",
    ["feedback_type"],
)


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus metrics HTTP server on a separate port."""
    try:
        start_http_server(port)
        logger.info("Prometheus metrics server started on port %d", port)
    except OSError as e:
        logger.warning("Could not start metrics server on port %d: %s", port, e)
