"""SQLite-backed review store for classification snapshots and overrides."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _migrate_legacy_references(conn: sqlite3.Connection) -> None:
    """Copy legacy per-file references into the normalized catalog/link tables."""
    legacy_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'classification_references'",
    ).fetchone()
    if not legacy_exists:
        return

    rows = conn.execute(
        """
        SELECT reviews.file_key,
               COALESCE(reviews.reviewed_antenna_type, reviews.auto_antenna_type, 'unknown') AS antenna_type,
               legacy.ordinal,
               legacy.url,
               legacy.title,
               legacy.note
        FROM classification_reviews AS reviews
        JOIN classification_references AS legacy
          ON legacy.file_key = reviews.file_key
        ORDER BY reviews.file_key, legacy.ordinal
        """
    ).fetchall()
    for file_key, antenna_type, ordinal, url, title, note in rows:
        if not url:
            continue
        conn.execute(
            """
            INSERT INTO antenna_type_references (antenna_type, url, title, note)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(antenna_type, url) DO UPDATE SET
                title = COALESCE(NULLIF(excluded.title, ''), antenna_type_references.title),
                note = COALESCE(NULLIF(excluded.note, ''), antenna_type_references.note),
                updated_at = CURRENT_TIMESTAMP
            """,
            (antenna_type, url, title, note),
        )
        ref_row = conn.execute(
            "SELECT id FROM antenna_type_references WHERE antenna_type = ? AND url = ?",
            (antenna_type, url),
        ).fetchone()
        if ref_row is None:
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO classification_reference_links (
                file_key, reference_id, ordinal
            ) VALUES (?, ?, ?)
            """,
            (file_key, int(ref_row[0]), int(ordinal or 0)),
        )


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS antenna_type_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                antenna_type TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (antenna_type, url)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classification_reference_links (
                file_key TEXT NOT NULL,
                reference_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (file_key, reference_id),
                FOREIGN KEY (file_key) REFERENCES classification_reviews(file_key)
                    ON DELETE CASCADE,
                FOREIGN KEY (reference_id) REFERENCES antenna_type_references(id)
                    ON DELETE CASCADE
            )
            """
        )
        _migrate_legacy_references(conn)
    return path


def list_type_references(db_path: str | Path, antenna_type: str) -> list[dict[str, Any]]:
    """List catalog references for a specific antenna type."""
    path = ensure_schema(db_path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, antenna_type, url, title, note
            FROM antenna_type_references
            WHERE antenna_type = ?
            ORDER BY COALESCE(NULLIF(title, ''), url), id
            """,
            (antenna_type,),
        ).fetchall()
    return [dict(row) for row in rows]


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
    antenna_type: str,
    reviewed_antenna_type: str | None,
    reason: str = "",
    reference_ids: list[int] | None = None,
    references: list[dict[str, str]] | None = None,
) -> None:
    """Persist a human review override and its supporting references."""
    path = ensure_schema(db_path)
    references = references or []
    reference_ids = reference_ids or []
    clean_type = reviewed_antenna_type.strip() if reviewed_antenna_type else None
    clean_type = clean_type or None
    catalog_type = antenna_type.strip() or clean_type or "unknown"
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
        conn.execute(
            "DELETE FROM classification_reference_links WHERE file_key = ?",
            (file_key,),
        )
        ordinal = 0
        linked_ids: set[int] = set()
        for ref_id in reference_ids:
            row = conn.execute(
                "SELECT id FROM antenna_type_references WHERE id = ? AND antenna_type = ?",
                (int(ref_id), catalog_type),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown reference id {ref_id} for antenna type {catalog_type}")
            linked_ids.add(int(ref_id))
            conn.execute(
                """
                INSERT INTO classification_reference_links (file_key, reference_id, ordinal)
                VALUES (?, ?, ?)
                """,
                (file_key, int(ref_id), ordinal),
            )
            ordinal += 1
        for index, ref in enumerate(references, start=ordinal):
            url = (ref.get("url") or "").strip()
            if not url:
                continue
            title = (ref.get("title") or "").strip() or None
            note = (ref.get("note") or "").strip() or None
            conn.execute(
                """
                INSERT INTO antenna_type_references (
                    antenna_type, url, title, note
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(antenna_type, url) DO UPDATE SET
                    title = COALESCE(NULLIF(excluded.title, ''), antenna_type_references.title),
                    note = COALESCE(NULLIF(excluded.note, ''), antenna_type_references.note),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (catalog_type, url, title, note),
            )
            ref_row = conn.execute(
                "SELECT id FROM antenna_type_references WHERE antenna_type = ? AND url = ?",
                (catalog_type, url),
            ).fetchone()
            if ref_row is None:
                continue
            ref_id = int(ref_row[0])
            if ref_id in linked_ids:
                continue
            linked_ids.add(ref_id)
            conn.execute(
                """
                INSERT INTO classification_reference_links (
                    file_key, reference_id, ordinal
                ) VALUES (?, ?, ?)
                """,
                (file_key, ref_id, index),
            )
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
                    title,
                    note,
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
            """
            SELECT catalog.id AS reference_id,
                   catalog.antenna_type,
                   catalog.url,
                   catalog.title,
                   catalog.note
            FROM classification_reference_links AS links
            JOIN antenna_type_references AS catalog
              ON catalog.id = links.reference_id
            WHERE links.file_key = ?
            ORDER BY links.ordinal
            """,
            (file_key,),
        ).fetchall()
        if not refs:
            refs = conn.execute(
                "SELECT NULL AS reference_id, NULL AS antenna_type, url, title, note FROM classification_references WHERE file_key = ? ORDER BY ordinal",
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