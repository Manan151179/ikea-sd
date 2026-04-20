"""
db.py — SQLite helper: single source of truth linking prompts, images, metrics.

Schema
------
runs
    run_id               INTEGER  PRIMARY KEY AUTOINCREMENT
    scene_id             TEXT
    room_type            TEXT
    strategy             TEXT     -- A / B / C / D
    control_mode         TEXT     -- none / mlsd
    seed                 INTEGER
    positive_prompt      TEXT
    negative_prompt      TEXT
    image_path           TEXT     UNIQUE
    reference_image_path TEXT
    created_at           TEXT     -- ISO-8601 UTC

metrics
    run_id               INTEGER  REFERENCES runs(run_id)
    metric_name          TEXT     -- clip_score / lpips / diversity / aesthetic
    value                REAL
    PRIMARY KEY (run_id, metric_name)

CLI
---
    python -m src.db --init <path>   # create the DB file and tables
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id             TEXT    NOT NULL,
    room_type            TEXT    NOT NULL,
    strategy             TEXT    NOT NULL,
    control_mode         TEXT    NOT NULL,
    seed                 INTEGER NOT NULL,
    positive_prompt      TEXT    NOT NULL,
    negative_prompt      TEXT,
    image_path           TEXT    UNIQUE NOT NULL,
    reference_image_path TEXT,
    created_at           TEXT    NOT NULL
);
"""

_DDL_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    metric_name TEXT    NOT NULL,
    value       REAL    NOT NULL,
    PRIMARY KEY (run_id, metric_name)
);
"""

# Required and optional fields for a runs row; used for validation in insert_run.
_RUNS_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "scene_id",
        "room_type",
        "strategy",
        "control_mode",
        "seed",
        "positive_prompt",
        "image_path",
    }
)
_RUNS_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {"negative_prompt", "reference_image_path"}
)
_RUNS_ALL_FIELDS: frozenset[str] = _RUNS_REQUIRED_FIELDS | _RUNS_OPTIONAL_FIELDS


class MissingRunFieldError(Exception):
    """Raised when a required runs field is absent from insert_run kwargs."""


class UnknownRunFieldError(Exception):
    """Raised when an unrecognised field name is passed to insert_run."""


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------


def open_connection(path: str | Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database at *path* and return the connection.

    WAL mode is enabled for better concurrent read performance.  Row factory is
    set to :class:`sqlite3.Row` so callers can access columns by name.

    Args:
        path: File-system path to the ``.db`` file.  The parent directory must
            already exist.

    Returns:
        Open :class:`sqlite3.Connection`.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def init_db(path: str | Path) -> sqlite3.Connection:
    """Create the database file at *path*, apply the schema, and return the connection.

    Safe to call on an existing database — uses ``CREATE TABLE IF NOT EXISTS``.

    Args:
        path: Destination ``.db`` file path.  Parent directory is created if it
            does not exist.

    Returns:
        Open :class:`sqlite3.Connection` to the initialised database.
    """
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = open_connection(db_path)
    with conn:
        conn.execute(_DDL_RUNS)
        conn.execute(_DDL_METRICS)

    logger.info("Database initialised at %s", db_path.resolve())
    return conn


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def insert_run(conn: sqlite3.Connection, **fields: Any) -> int:
    """Insert one row into ``runs`` and return the new ``run_id``.

    A UTC ``created_at`` timestamp is injected automatically; callers must not
    pass it.

    Args:
        conn: Open database connection.
        **fields: Column values for the ``runs`` table.  All keys in
            :data:`_RUNS_REQUIRED_FIELDS` are mandatory.  Optional keys are
            ``negative_prompt`` and ``reference_image_path`` (default
            ``None``).

    Returns:
        The auto-assigned ``run_id`` integer.

    Raises:
        MissingRunFieldError: If any required field is absent.
        UnknownRunFieldError: If an unrecognised field name is passed.
    """
    unknown = set(fields) - _RUNS_ALL_FIELDS
    if unknown:
        raise UnknownRunFieldError(
            f"Unrecognised field(s) for runs table: {sorted(unknown)}"
        )

    missing = _RUNS_REQUIRED_FIELDS - set(fields)
    if missing:
        raise MissingRunFieldError(
            f"Required field(s) missing for runs table: {sorted(missing)}"
        )

    row = {
        "scene_id": fields["scene_id"],
        "room_type": fields["room_type"],
        "strategy": fields["strategy"],
        "control_mode": fields["control_mode"],
        "seed": fields["seed"],
        "positive_prompt": fields["positive_prompt"],
        "negative_prompt": fields.get("negative_prompt"),
        "image_path": fields["image_path"],
        "reference_image_path": fields.get("reference_image_path"),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    sql = """
        INSERT INTO runs (
            scene_id, room_type, strategy, control_mode, seed,
            positive_prompt, negative_prompt,
            image_path, reference_image_path, created_at
        ) VALUES (
            :scene_id, :room_type, :strategy, :control_mode, :seed,
            :positive_prompt, :negative_prompt,
            :image_path, :reference_image_path, :created_at
        )
    """
    with conn:
        cursor = conn.execute(sql, row)

    run_id: int = cursor.lastrowid  # type: ignore[assignment]
    logger.debug("Inserted run_id=%d for scene_id=%s", run_id, row["scene_id"])
    return run_id


def insert_metric(
    conn: sqlite3.Connection,
    run_id: int,
    name: str,
    value: float,
) -> None:
    """Insert or replace one metric value for a given run.

    Uses ``INSERT OR REPLACE`` so re-scoring a run overwrites the previous
    value rather than raising a constraint error.

    Args:
        conn: Open database connection.
        run_id: The ``run_id`` this metric belongs to.
        name: Metric identifier, e.g. ``"clip_score"``, ``"lpips"``.
        value: Numeric metric value.
    """
    sql = """
        INSERT OR REPLACE INTO metrics (run_id, metric_name, value)
        VALUES (?, ?, ?)
    """
    with conn:
        conn.execute(sql, (run_id, name, value))

    logger.debug("Inserted metric run_id=%d  %s=%.4f", run_id, name, value)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def query_df(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[Any, ...] = (),
) -> pd.DataFrame:
    """Execute *sql* with *params* and return results as a :class:`pandas.DataFrame`.

    Intended for analysis notebooks and ad-hoc queries; not for hot paths.

    Args:
        conn: Open database connection.
        sql: Parameterised SQL query string.
        params: Positional query parameters (use ``?`` placeholders).

    Returns:
        Query result as a :class:`~pandas.DataFrame`.  Empty DataFrame if no
        rows match.
    """
    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    if not rows:
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        return pd.DataFrame(columns=col_names)
    col_names = [d[0] for d in cursor.description]
    return pd.DataFrame([dict(row) for row in rows], columns=col_names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialise the ikea-sd SQLite database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--init",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path at which to create (or verify) the database file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(name)s  %(message)s",
    )

    conn = init_db(args.init)
    conn.close()
    print(f"Database ready: {Path(args.init).resolve()}")
