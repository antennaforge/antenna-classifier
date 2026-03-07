"""Postgres storage for user-generated antennas.

Connects to the hamfeeds wspr Postgres (reachable via Docker network).
Stores antenna metadata + full NEC content in an ``ac_user_antennas`` table.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_DB_CONFIG: dict[str, Any] = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "wspr"),
    "user": os.getenv("POSTGRES_USER", "wspr_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "change_this_password_in_production"),
}


def _conn():
    return psycopg2.connect(**_DB_CONFIG)


def ensure_table() -> None:
    """Create the ac_user_antennas table if it doesn't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS ac_user_antennas (
        id            SERIAL PRIMARY KEY,
        name          VARCHAR(255) NOT NULL,
        description   TEXT DEFAULT '',
        antenna_type  VARCHAR(100),
        frequency_mhz DOUBLE PRECISION,
        band          VARCHAR(20),
        ground_type   VARCHAR(50) DEFAULT 'free_space',
        nec_content   TEXT NOT NULL,
        source        VARCHAR(20) DEFAULT 'form',
        metadata      JSONB DEFAULT '{}',
        created_at    TIMESTAMPTZ DEFAULT NOW(),
        updated_at    TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_ac_user_antennas_type
        ON ac_user_antennas(antenna_type);
    CREATE INDEX IF NOT EXISTS idx_ac_user_antennas_created
        ON ac_user_antennas(created_at DESC);
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def list_antennas(limit: int = 200) -> list[dict]:
    """Return all user antennas (newest first), without NEC content."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT id, name, description, antenna_type, frequency_mhz,
                          band, ground_type, source,
                          created_at, updated_at
                   FROM ac_user_antennas
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (limit,),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_antenna(antenna_id: int) -> dict | None:
    """Fetch a single antenna including NEC content."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM ac_user_antennas WHERE id = %s", (antenna_id,)
            )
            row = cur.fetchone()
    return dict(row) if row else None


def create_antenna(
    *,
    name: str,
    description: str = "",
    antenna_type: str | None = None,
    frequency_mhz: float | None = None,
    band: str | None = None,
    ground_type: str = "free_space",
    nec_content: str,
    source: str = "form",
    metadata: dict | None = None,
) -> dict:
    """Insert a new user antenna and return it."""
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO ac_user_antennas
                       (name, description, antenna_type, frequency_mhz,
                        band, ground_type, nec_content, source, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING *""",
                (
                    name,
                    description,
                    antenna_type,
                    frequency_mhz,
                    band,
                    ground_type,
                    nec_content,
                    source,
                    json.dumps(metadata or {}),
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return dict(row)


def update_antenna(antenna_id: int, **fields) -> dict | None:
    """Update mutable fields on a user antenna."""
    allowed = {
        "name", "description", "antenna_type", "frequency_mhz",
        "band", "ground_type", "nec_content", "metadata",
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return get_antenna(antenna_id)
    if "metadata" in updates and isinstance(updates["metadata"], dict):
        updates["metadata"] = json.dumps(updates["metadata"])
    set_clause = ", ".join(f"{k} = %s" for k in updates)
    values = list(updates.values()) + [antenna_id]
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"UPDATE ac_user_antennas SET {set_clause}, updated_at = NOW() "
                f"WHERE id = %s RETURNING *",
                values,
            )
            row = cur.fetchone()
        conn.commit()
    return dict(row) if row else None


def delete_antenna(antenna_id: int) -> bool:
    """Delete a user antenna. Returns True if a row was deleted."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM ac_user_antennas WHERE id = %s", (antenna_id,)
            )
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted
