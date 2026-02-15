"""Optional LangSmith tracing configuration."""

import logging
import os

logger = logging.getLogger(__name__)


def configure_langsmith(
    api_key: str = "", project: str = "s7-1200-rag"
) -> None:
    """Set LangSmith env vars if api_key is provided. No-op otherwise."""
    if not api_key:
        logger.info("LangSmith tracing disabled (no API key)")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    logger.info("LangSmith tracing enabled for project '%s'", project)
