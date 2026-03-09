"""SQLite-backed review store for classification snapshots and overrides."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def ensure_schema(db_path: str | Path) -> Path:
    """Create the SQLite schema if it does not already exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classification_reviews (
                file_key TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                path TEXT NOT NULL,
                valid INTEGER NOT NULL DEFAULT 0,
                auto_antenna_type TEXT NOT NULL,
                auto_confidence REAL NOT NULL DEFAULT 0,
                auto_frequency_mhz REAL,
                auto_band TEXT,
                auto_element_count INTEGER NOT NULL DEFAULT 0,
                auto_ground_type TEXT,
                auto_wire_count INTEGER NOT NULL DEFAULT 0,
                fingerprint_signature TEXT,
                fingerprint_json TEXT,
                complexity REAL NOT NULL DEFAULT 0,
                evidence_json TEXT,
                subtypes_json TEXT,
                reviewed_antenna_type TEXT,
                review_reason TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classification_references (
                file_key TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (file_key, ordinal),
                FOREIGN KEY (file_key) REFERENCES classification_reviews(file_key)
                    ON DELETE CASCADE
            )
            """
        )
    return path


def upsert_auto_record(db_path: str | Path, record: dict[str, Any]) -> None:
    """Persist the latest automatic classification snapshot for a file."""
    path = ensure_schema(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO classification_reviews (
                file_key, filename, relative_path, path, valid,
                auto_antenna_type, auto_confidence, auto_frequency_mhz,
                auto_band, auto_element_count, auto_ground_type,
                auto_wire_count, fingerprint_signature, fingerprint_json,
                complexity, evidence_json, subtypes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_key) DO UPDATE SET
                filename = excluded.filename,
                relative_path = excluded.relative_path,
                path = excluded.path,
                valid = excluded.valid,
                auto_antenna_type = excluded.auto_antenna_type,
                auto_confidence = excluded.auto_confidence,
                auto_frequency_mhz = excluded.auto_frequency_mhz,
                auto_band = excluded.auto_band,
                auto_element_count = excluded.auto_element_count,
                auto_ground_type = excluded.auto_ground_type,
                auto_wire_count = excluded.auto_wire_count,
                fingerprint_signature = excluded.fingerprint_signature,
                fingerprint_json = excluded.fingerprint_json,
                complexity = excluded.complexity,
                evidence_json = excluded.evidence_json,
                subtypes_json = excluded.subtypes_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                record["file_key"],
                record["filename"],
                record["relative_path"],
                record["path"],
                1 if record.get("valid") else 0,
                record.get("auto_antenna_type") or record.get("antenna_type") or "unknown",
                float(record.get("confidence") or 0.0),
                record.get("frequency_mhz"),
                record.get("band"),
                int(record.get("element_count") or 0),
                record.get("ground_type"),
                int(record.get("wire_count") or 0),
                record.get("fingerprint"),
                json.dumps(record.get("fingerprint_details") or {}, sort_keys=True),
                float(record.get("complexity") or 0.0),
                json.dumps(record.get("evidence") or []),
                json.dumps(record.get("subtypes") or []),
            ),
        )


def save_review(
    db_path: str | Path,
    file_key: str,
    *,
    reviewed_antenna_type: str | None,
    reason: str = "",
    references: list[dict[str, str]] | None = None,
) -> None:
    """Persist a human review override and its supporting references."""
    path = ensure_schema(db_path)
    references = references or []
    clean_type = reviewed_antenna_type.strip() if reviewed_antenna_type else None
    clean_type = clean_type or None
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        updated = conn.execute(
            """
            UPDATE classification_reviews
            SET reviewed_antenna_type = ?,
                review_reason = ?,
                reviewed_at = CASE WHEN ? IS NULL AND ? = '' THEN NULL ELSE CURRENT_TIMESTAMP END,
                updated_at = CURRENT_TIMESTAMP
            WHERE file_key = ?
            """,
            (clean_type, reason.strip(), clean_type, reason.strip(), file_key),
        )
        if updated.rowcount == 0:
            raise KeyError(file_key)
        conn.execute(
            "DELETE FROM classification_references WHERE file_key = ?",
            (file_key,),
        )
        for index, ref in enumerate(references):
            url = (ref.get("url") or "").strip()
            if not url:
                continue
            conn.execute(
                """
                INSERT INTO classification_references (
                    file_key, ordinal, url, title, note
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file_key,
                    index,
                    url,
                    (ref.get("title") or "").strip() or None,
                    (ref.get("note") or "").strip() or None,
                ),
            )


def get_record(db_path: str | Path, file_key: str) -> dict[str, Any] | None:
    """Load the stored review snapshot for a file."""
    path = Path(db_path)
    if not path.exists():
        return None
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM classification_reviews WHERE file_key = ?",
            (file_key,),
        ).fetchone()
        if row is None:
            return None
        refs = conn.execute(
            "SELECT url, title, note FROM classification_references WHERE file_key = ? ORDER BY ordinal",
            (file_key,),
        ).fetchall()
    return {
        "file_key": row["file_key"],
        "filename": row["filename"],
        "relative_path": row["relative_path"],
        "path": row["path"],
        "valid": bool(row["valid"]),
        "auto_antenna_type": row["auto_antenna_type"],
        "auto_confidence": float(row["auto_confidence"] or 0.0),
        "auto_frequency_mhz": row["auto_frequency_mhz"],
        "auto_band": row["auto_band"],
        "auto_element_count": int(row["auto_element_count"] or 0),
        "auto_ground_type": row["auto_ground_type"],
        "auto_wire_count": int(row["auto_wire_count"] or 0),
        "fingerprint_signature": row["fingerprint_signature"],
        "fingerprint_details": json.loads(row["fingerprint_json"] or "{}"),
        "complexity": float(row["complexity"] or 0.0),
        "evidence": json.loads(row["evidence_json"] or "[]"),
        "subtypes": json.loads(row["subtypes_json"] or "[]"),
        "reviewed_antenna_type": row["reviewed_antenna_type"],
        "review_reason": row["review_reason"] or "",
        "reviewed_at": row["reviewed_at"],
        "updated_at": row["updated_at"],
        "references": [dict(ref) for ref in refs],
    }


def merge_review(auto_record: dict[str, Any], stored: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a live auto-classification snapshot with persisted review data."""
    merged = dict(auto_record)
    merged["auto_antenna_type"] = auto_record.get("antenna_type")
    merged["fingerprint_details"] = auto_record.get("fingerprint_details") or {}
    merged["is_reviewed"] = False
    merged["reviewed_antenna_type"] = None
    merged["review_reason"] = ""
    merged["reviewed_at"] = None
    merged["references"] = []
    merged["effective_antenna_type"] = auto_record.get("antenna_type")
    if stored is None:
        return merged
    merged["is_reviewed"] = bool(stored.get("reviewed_antenna_type") or stored.get("review_reason") or stored.get("references"))
    merged["reviewed_antenna_type"] = stored.get("reviewed_antenna_type")
    merged["review_reason"] = stored.get("review_reason") or ""
    merged["reviewed_at"] = stored.get("reviewed_at")
    merged["references"] = stored.get("references") or []
    merged["fingerprint_details"] = stored.get("fingerprint_details") or merged["fingerprint_details"]
    merged["effective_antenna_type"] = stored.get("reviewed_antenna_type") or auto_record.get("antenna_type")
    merged["antenna_type"] = merged["effective_antenna_type"]
    return merged