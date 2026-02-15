"""RAG chain implementation for S7-1200 chatbot.

Orchestrates retrieval, prompt construction, LLM generation,
citation extraction, and conversation memory.
"""

import json
import logging
import re
import time
from typing import Generator, Optional

from langchain_community.llms import Ollama
from langchain.schema import Document

from src.config import settings
from src.retrieval.vector_store import QdrantVectorStore
from src.monitoring.rag_quality import RAGQualityTracker
from src.monitoring import metrics as mon

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert technical assistant for Siemens S7-1200 Programmable Logic Controllers.

CORE REQUIREMENTS:
1. ALWAYS cite page numbers: [Page X] or [Chapter Y, Page Z]
2. ALWAYS highlight safety warnings with ⚠️
3. Provide complete, actionable answers
4. Use exact technical terminology from Siemens documentation
5. Include verification steps when applicable

RESPONSE STRUCTURE:

For Configuration Questions:
- Hardware requirements [Page X]
- Numbered TIA Portal steps
- Parameter settings table
- Code example (if needed)
- Verification procedure
- Citations

For Troubleshooting:
- Problem confirmation
- Diagnostic steps
- Common causes (ranked)
- Solutions
- Prevention
- Citations

For Programming:
- Concept explanation
- Code with comments
- Parameter details
- Error handling
- Performance notes
- Citations

CODE FORMAT:
Use properly formatted code blocks with language tag (scl, ladder, etc.) and inline comments.
Always include the source page reference after code blocks.

SAFETY WARNINGS:
When the context contains safety-critical information (WARNING, CAUTION, DANGER), always present it as:
⚠️ WARNING: [Hazard description]
- Consequence: [What could happen]
- Action: [Required safety measures]

WHEN INFORMATION IS MISSING:
If the retrieved context does not contain enough information to fully answer the question, say:
"Based on the retrieved sections, I don't have specific information about [topic].
⚠️ Verify with the official Siemens documentation before implementation."

Be thorough, precise, and professional. Engineers rely on your accuracy for critical industrial systems.
"""

USER_PROMPT_TEMPLATE = """\
Retrieved Context from S7-1200 Manual:
{context}

Conversation History:
{chat_history}

User Question: {question}

Provide a comprehensive answer following the system guidelines. Include page references from the retrieved context.
"""


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents with source metadata for LLM context."""
    if not docs:
        return "No relevant documents found."

    formatted: list[str] = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "N/A")
        chapter = doc.metadata.get("chapter", "")
        section = doc.metadata.get("section", "")
        location = section if section else chapter
        formatted.append(
            f"--- Source {i} (Page {page}, {location}) ---\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def format_chat_history(history: list[tuple[str, str]], max_turns: int = 10) -> str:
    """Format conversation history for inclusion in the prompt."""
    if not history:
        return "No previous conversation."

    recent = history[-max_turns:]
    lines: list[str] = []
    for user_msg, assistant_msg in recent:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    return "\n".join(lines)


def extract_citations(text: str) -> list[str]:
    """Extract page/chapter citations from generated text."""
    patterns = [
        r"\[Page\s+(\d+)\]",
        r"\[Chapter\s+[\d.]+,?\s*Page\s+(\d+)\]",
        r"\[Pages?\s+(\d+(?:\s*[-–]\s*\d+)?)\]",
        r"Page\s+(\d+)",
    ]
    citations: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            cite = match.group(0)
            if cite not in seen:
                seen.add(cite)
                citations.append(cite)
    return citations


class S7RAGChain:
    """Conversational RAG chain for S7-1200 technical questions."""

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.temperature = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        self.top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        self.conversation_history: list[tuple[str, str]] = []
        self.quality_tracker = RAGQualityTracker(
            low_score_threshold=settings.LOW_SCORE_THRESHOLD
        )

        self.llm = Ollama(
            base_url=settings.OLLAMA_URL,
            model=settings.OLLAMA_MODEL,
            temperature=self.temperature,
            top_p=settings.OLLAMA_TOP_P,
            num_ctx=settings.OLLAMA_NUM_CTX,
            repeat_penalty=settings.OLLAMA_REPEAT_PENALTY,
        )
        logger.info(
            "RAG chain initialized: model=%s, temperature=%.2f, top_k=%d",
            settings.OLLAMA_MODEL, self.temperature, self.top_k,
        )

    def update_settings(self, temperature: Optional[float] = None, top_k: Optional[int] = None) -> None:
        """Dynamically update chain parameters."""
        if temperature is not None:
            self.temperature = temperature
            self.llm.temperature = temperature
        if top_k is not None:
            self.top_k = top_k

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant documents for the query."""
        return self.vector_store.hybrid_search(query, k=self.top_k)

    def generate(
        self,
        query: str,
        retrieved_docs: Optional[list[Document]] = None,
    ) -> dict:
        """Run the full RAG pipeline: retrieve -> format -> generate -> extract citations.

        Args:
            query: User question.
            retrieved_docs: Pre-retrieved documents (if None, retrieval runs automatically).

        Returns:
            Dict with keys: answer, sources, citations, metadata.
        """
        start_time = time.time()

        # Retrieve
        if retrieved_docs is None:
            retrieved_docs = self.retrieve(query)
        retrieval_time = time.time() - start_time

        # Format context
        context = format_docs(retrieved_docs)
        history_text = format_chat_history(self.conversation_history)

        # Build prompt
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(context=context, chat_history=history_text, question=query)}"

        # Generate
        gen_start = time.time()
        llm_failed = False
        try:
            answer = self.llm.invoke(prompt)
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            mon.LLM_ERRORS.labels(error_type=type(e).__name__).inc()
            mon.REQUEST_COUNT.labels(status="error").inc()
            llm_failed = True
            answer = (
                f"I'm unable to generate a response at this time. "
                f"Error: {e}\n\n"
                f"Please ensure Ollama is running with the model '{settings.OLLAMA_MODEL}'."
            )
        generation_time = time.time() - gen_start
        total_time = time.time() - start_time

        # Extract citations
        citations = extract_citations(answer)

        # Quality evaluation
        quality_metrics = self.quality_tracker.evaluate_retrieval(query, retrieved_docs)
        retrieved_pages = [doc.metadata.get("page", 0) for doc in retrieved_docs]
        citation_metrics = self.quality_tracker.evaluate_citations(citations, retrieved_pages)

        # Approximate token throughput
        approx_tokens = len(answer.split()) / 0.75
        tokens_per_sec = approx_tokens / generation_time if generation_time > 0 else 0

        # Record Prometheus metrics
        if not llm_failed:
            mon.REQUEST_COUNT.labels(status="success").inc()
        mon.RESPONSE_TIME.observe(total_time)
        mon.RETRIEVAL_TIME.observe(retrieval_time)
        mon.GENERATION_TIME.observe(generation_time)
        mon.RETRIEVAL_SCORE.observe(quality_metrics["avg_relevance_score"])
        mon.CITATION_ACCURACY.observe(citation_metrics["citation_accuracy"])
        if quality_metrics["low_quality_retrieval"]:
            mon.LOW_QUALITY_RETRIEVALS.inc()
        if tokens_per_sec > 0:
            mon.TOKENS_PER_SECOND.observe(tokens_per_sec)

        # Update conversation history
        self.conversation_history.append((query, answer))
        if len(self.conversation_history) > settings.MAX_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-settings.MAX_HISTORY_LENGTH:]

        return {
            "answer": answer,
            "sources": retrieved_docs,
            "citations": citations,
            "metadata": {
                "query": query,
                "num_sources": len(retrieved_docs),
                "citations_found": len(citations),
                "retrieval_time_s": round(retrieval_time, 3),
                "generation_time_s": round(generation_time, 3),
                "total_time_s": round(total_time, 3),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "model": settings.OLLAMA_MODEL,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "quality": quality_metrics,
                "citation_check": citation_metrics,
            },
        }

    def stream(self, query: str) -> Generator[str, None, None]:
        """Stream the RAG response token by token.

        Yields:
            Individual tokens/chunks from the LLM.
        """
        retrieved_docs = self.retrieve(query)
        context = format_docs(retrieved_docs)
        history_text = format_chat_history(self.conversation_history)

        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(context=context, chat_history=history_text, question=query)}"

        full_response = ""
        try:
            for chunk in self.llm.stream(prompt):
                full_response += chunk
                yield chunk
        except Exception as e:
            error_msg = f"\n\nStreaming error: {e}"
            full_response += error_msg
            yield error_msg

        # Update history after streaming completes
        self.conversation_history.append((query, full_response))
        if len(self.conversation_history) > settings.MAX_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-settings.MAX_HISTORY_LENGTH:]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_source_display(self, sources: list[Document]) -> str:
        """Format sources for display in the UI sidebar."""
        if not sources:
            return "No sources retrieved."

        parts: list[str] = []
        for i, doc in enumerate(sources, 1):
            page = doc.metadata.get("page", "N/A")
            chapter = doc.metadata.get("chapter", "")
            section = doc.metadata.get("section", "")
            score = doc.metadata.get("score", "N/A")
            topics = doc.metadata.get("topics", [])

            header = f"### Source {i} — Page {page}"
            if section:
                header += f"\n**{section}**"
            elif chapter:
                header += f"\n**{chapter}**"

            content_preview = doc.page_content[:300]
            if len(doc.page_content) > 300:
                content_preview += "..."

            meta_line = f"Score: {score}"
            if topics:
                meta_line += f" | Topics: {', '.join(topics)}"

            parts.append(f"{header}\n\n{content_preview}\n\n*{meta_line}*\n\n---")

        return "\n\n".join(parts)
