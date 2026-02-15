"""PDF processing pipeline for S7-1200 system manual.

Handles PDF parsing, smart chunking, and metadata extraction
for the 864-page Siemens S7-1200 technical manual.
"""

import re
import logging
from pathlib import Path
from typing import Optional

import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

# Patterns for content detection
CHAPTER_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*)\s+(.+)$", re.MULTILINE
)
SAFETY_PATTERN = re.compile(
    r"\b(WARNING|CAUTION|DANGER|NOTICE)\b", re.IGNORECASE
)
CODE_PATTERN = re.compile(
    r"(:=|;|\bIF\b|\bTHEN\b|\bEND_IF\b|\bFOR\b|\bWHILE\b|\bFUNCTION\b|\bFUNCTION_BLOCK\b"
    r"|\bVAR\b|\bEND_VAR\b|\bBEGIN\b|\bEND_FUNCTION\b|\bDATA_BLOCK\b)",
    re.IGNORECASE,
)
TABLE_PATTERN = re.compile(r"(\|.*\|)|(\t.*\t.*\t)")
SPEC_PATTERN = re.compile(
    r"(\d+\s*(V|mA|kHz|ms|µs|°C|mm|kg|W|A|Ω|bytes?|KB|MB))\b", re.IGNORECASE
)

# Topic keywords mapped to tags
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "analog_io": ["analog", "AI ", "AQ ", "ADC", "scaling", "4-20 mA", "0-10 V"],
    "digital_io": ["digital input", "digital output", "DI ", "DQ ", "24V DC"],
    "communication": [
        "PROFINET", "Ethernet", "TCP", "UDP", "S7 communication",
        "OPC", "Modbus", "PUT/GET", "Open User Communication",
    ],
    "programming": ["SCL", "LAD", "FBD", "STL", "program block", "function block", "OB"],
    "pid_control": ["PID", "PID_Compact", "PID_3Step", "setpoint", "process value"],
    "motion_control": ["motion", "axis", "MC_", "homing", "positioning", "pulse"],
    "safety": ["safety", "fail-safe", "F-CPU", "F-DI", "F-DQ", "SIL"],
    "hmi": ["HMI", "KTP", "panel", "visualization", "alarm", "display"],
    "hardware": ["CPU", "signal module", "signal board", "CM ", "CB ", "battery board"],
    "diagnostics": ["diagnostic", "error", "fault", "LED", "status", "troubleshoot"],
    "data_types": ["BOOL", "INT", "REAL", "DINT", "STRING", "ARRAY", "STRUCT", "UDT"],
    "timers_counters": ["timer", "counter", "TON", "TOF", "TP ", "CTU", "CTD"],
    "web_server": ["web server", "HTTP", "user-defined web", "standard web"],
    "clock": ["time-of-day", "clock", "RTC", "system time"],
    "memory": ["memory", "bit memory", "data block", "DB ", "work memory", "load memory"],
}


class S7ManualProcessor:
    """Processes the S7-1200 system manual PDF into LangChain Documents."""

    def __init__(self, pdf_path: Optional[Path] = None):
        self.pdf_path = pdf_path or settings.pdf_file
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n\n",   # Major section breaks
                "\n\n",     # Paragraph breaks
                "\n",       # Line breaks
                ". ",       # Sentences
                ", ",       # Clauses
                " ",        # Words
            ],
            length_function=len,
            is_separator_regex=False,
        )

    def extract_pages(self) -> list[dict]:
        """Extract text and metadata from every page of the PDF.

        Returns:
            List of dicts with keys: page_number, text, tables_count.
        """
        logger.info("Opening PDF: %s", self.pdf_path)
        pages: list[dict] = []

        with pdfplumber.open(str(self.pdf_path)) as pdf:
            total = len(pdf.pages)
            logger.info("Total pages: %d", total)

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text() or ""
                tables = page.extract_tables() or []

                # Append table content as formatted text
                table_texts: list[str] = []
                for table in tables:
                    rows = []
                    for row in table:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    table_texts.append("\n".join(rows))

                combined_text = text
                if table_texts:
                    combined_text += "\n\n" + "\n\n".join(table_texts)

                pages.append({
                    "page_number": page_num,
                    "text": combined_text,
                    "tables_count": len(tables),
                })

                if page_num % 100 == 0:
                    logger.info("Extracted %d / %d pages", page_num, total)

        logger.info("Extraction complete: %d pages", len(pages))
        return pages

    def _detect_chapter(self, text: str) -> str:
        """Detect chapter/section heading from text content."""
        lines = text.strip().split("\n")[:5]  # Check first 5 lines
        for line in lines:
            line = line.strip()
            match = CHAPTER_PATTERN.match(line)
            if match:
                section_num = match.group(1)
                section_title = match.group(2).strip()
                depth = section_num.count(".")
                if depth == 0:
                    return f"Chapter {section_num}: {section_title}"
                return f"Section {section_num}: {section_title}"
        return ""

    def _detect_topics(self, text: str) -> list[str]:
        """Detect topic tags based on keyword matching."""
        text_lower = text.lower()
        topics: list[str] = []
        for topic, keywords in TOPIC_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    topics.append(topic)
                    break
        return topics

    def _has_code(self, text: str) -> bool:
        """Detect if text contains PLC programming code."""
        matches = CODE_PATTERN.findall(text)
        return len(matches) >= 2

    def _has_table(self, text: str) -> bool:
        """Detect if text contains table-like structures."""
        return bool(TABLE_PATTERN.search(text))

    def _has_safety_warning(self, text: str) -> bool:
        """Detect safety-critical content."""
        return bool(SAFETY_PATTERN.search(text))

    def _has_specs(self, text: str) -> bool:
        """Detect technical specifications."""
        return bool(SPEC_PATTERN.search(text))

    def create_documents(self, pages: list[dict]) -> list[Document]:
        """Convert extracted pages into chunked LangChain Documents with metadata.

        Args:
            pages: Output from extract_pages().

        Returns:
            List of Document objects ready for embedding.
        """
        logger.info("Creating documents from %d pages", len(pages))

        # Track current chapter context across pages
        current_chapter = ""
        current_section = ""
        documents: list[Document] = []

        for page_info in pages:
            page_num = page_info["page_number"]
            text = page_info["text"]

            if not text.strip():
                continue

            # Update chapter/section tracking
            detected = self._detect_chapter(text)
            if detected:
                if detected.startswith("Chapter"):
                    current_chapter = detected
                    current_section = detected
                else:
                    current_section = detected

            # Chunk the page text
            chunks = self.text_splitter.split_text(text)

            for chunk in chunks:
                if len(chunk.strip()) < 30:
                    continue  # Skip tiny fragments

                metadata = {
                    "page": page_num,
                    "chapter": current_chapter,
                    "section": current_section,
                    "has_code": self._has_code(chunk),
                    "has_tables": self._has_table(chunk) or page_info["tables_count"] > 0,
                    "has_safety_warning": self._has_safety_warning(chunk),
                    "has_specs": self._has_specs(chunk),
                    "topics": self._detect_topics(chunk),
                    "source": str(self.pdf_path.name),
                }

                documents.append(Document(page_content=chunk, metadata=metadata))

        logger.info("Created %d document chunks", len(documents))
        return documents

    def process(self) -> list[Document]:
        """Run the full processing pipeline: extract -> chunk -> enrich.

        Returns:
            List of enriched Document objects.
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found at {self.pdf_path}. "
                "Place the S7-1200 manual PDF in data/raw/ or set PDF_PATH in .env"
            )

        pages = self.extract_pages()
        documents = self.create_documents(pages)

        # Log statistics
        code_docs = sum(1 for d in documents if d.metadata["has_code"])
        safety_docs = sum(1 for d in documents if d.metadata["has_safety_warning"])
        table_docs = sum(1 for d in documents if d.metadata["has_tables"])
        topic_counts: dict[str, int] = {}
        for d in documents:
            for t in d.metadata["topics"]:
                topic_counts[t] = topic_counts.get(t, 0) + 1

        logger.info("--- Processing Statistics ---")
        logger.info("Total chunks: %d", len(documents))
        logger.info("Chunks with code: %d", code_docs)
        logger.info("Chunks with safety warnings: %d", safety_docs)
        logger.info("Chunks with tables: %d", table_docs)
        logger.info("Topic distribution: %s", topic_counts)

        return documents
