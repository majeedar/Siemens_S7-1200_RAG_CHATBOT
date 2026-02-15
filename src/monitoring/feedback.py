"""User feedback storage using SQLite."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Stores user feedback in a local SQLite database."""

    def __init__(self, db_path: Optional[Path] = None):
        from src.config import settings

        self.db_path = db_path or (settings.data_dir / "feedback.db")
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    comment TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.commit()
        logger.info("Feedback database initialized at %s", self.db_path)

    def record(
        self,
        query: str,
        answer: str,
        feedback_type: str,
        comment: str = "",
        metadata: Optional[dict] = None,
    ) -> int:
        """Record a feedback event. Returns the row ID."""
        from src.monitoring.metrics import FEEDBACK_COUNT

        FEEDBACK_COUNT.labels(feedback_type=feedback_type).inc()

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """INSERT INTO feedback
                   (timestamp, query, answer, feedback_type, comment, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    query,
                    answer,
                    feedback_type,
                    comment,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
            logger.info(
                "Feedback recorded: type=%s id=%d", feedback_type, cursor.lastrowid
            )
            return cursor.lastrowid

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Retrieve recent feedback entries."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Get feedback summary statistics."""
        with sqlite3.connect(str(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            by_type = conn.execute(
                "SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type"
            ).fetchall()
            return {
                "total": total,
                "by_type": {row[0]: row[1] for row in by_type},
            }
