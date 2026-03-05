"""Memory Domain — sqlite-vec Repository Adapter.

Implements the MemoryRepository protocol using sqlite-vec for vector search.
Lazy connection: the database and tables are created on first use.
All operations are wrapped in try/except to ensure failures never propagate.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..entities import MemoryRecord
from ..value_objects import (
    EmbeddingVector,
    MemoryEntry,
    MemoryType,
    SimilarityScore,
)

logger = logging.getLogger(__name__)


def _pack_vector(values: Tuple[float, ...]) -> bytes:
    """Pack a float tuple into raw bytes for sqlite-vec vec0 tables."""
    return struct.pack(f"{len(values)}f", *values)


def _sanitize_collection_name(name: str) -> str:
    """Ensure collection name is safe for use as a SQL identifier.

    Only allows alphanumeric characters and underscores.
    """
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = "c_" + sanitized
    return sanitized


class SqliteVecRepository:
    """MemoryRepository implementation backed by sqlite-vec.

    Uses two tables per collection:
      - ``{name}_vec``: A vec0 virtual table holding the embedding vectors.
      - ``{name}_meta``: A regular table holding record metadata, linked to
        vec rows via ``vec_rowid``.

    Thread safety: sqlite3 connections are *not* shared across threads.
    Each call to ``_get_connection()`` returns a module-level connection that
    is created lazily on first use. For multi-threaded use, callers should
    instantiate one repository per thread or add external synchronisation.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        # Track which collections have been initialised this session so we
        # don't re-run CREATE TABLE on every operation.
        self._initialised_collections: set[str] = set()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return (and lazily create) the SQLite connection.

        On first call this will:
        1. Create parent directories for the DB file.
        2. Open the connection.
        3. Load the sqlite-vec extension.
        4. Enable WAL journal mode for better concurrent read performance.
        """
        if self._conn is not None:
            return self._conn

        db_dir = Path(self._db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self._db_path)
        conn.enable_load_extension(True)

        import sqlite_vec  # type: ignore[import-untyped]

        sqlite_vec.load(conn)

        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row

        self._conn = conn
        return conn

    def close(self) -> None:
        """Close the underlying connection if open."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing sqlite-vec connection", exc_info=True)
            finally:
                self._conn = None
                self._initialised_collections.clear()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _ensure_tables(self, collection_name: str, dimension: int) -> None:
        """Create the vec0 virtual table and metadata table if needed."""
        safe_name = _sanitize_collection_name(collection_name)
        if safe_name in self._initialised_collections:
            return

        conn = self._get_connection()

        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS [{safe_name}_vec] "
            f"USING vec0(embedding float[{int(dimension)}])"
        )

        conn.execute(
            f"CREATE TABLE IF NOT EXISTS [{safe_name}_meta] ("
            "  record_id TEXT PRIMARY KEY,"
            "  vec_rowid INTEGER NOT NULL,"
            "  content TEXT NOT NULL,"
            "  memory_type TEXT NOT NULL,"
            "  metadata_json TEXT,"
            "  tags_json TEXT,"
            "  session_id TEXT,"
            "  created_at TEXT NOT NULL,"
            "  accessed_at TEXT NOT NULL,"
            "  access_count INTEGER DEFAULT 0"
            ")"
        )

        conn.commit()
        self._initialised_collections.add(safe_name)

    def _get_dimension(self, collection_name: str) -> Optional[int]:
        """Attempt to read the dimension of an existing vec0 table.

        Returns None if the table does not exist or the query fails.
        """
        safe_name = _sanitize_collection_name(collection_name)
        try:
            conn = self._get_connection()
            # vec0 tables expose column info via a special shadow table.
            # A simpler way: insert nothing but parse the schema.
            row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (f"{safe_name}_vec",),
            ).fetchone()
            if row is None:
                return None
            sql: str = row[0] if isinstance(row, tuple) else row["sql"]
            # Pattern: float[256]
            start = sql.find("float[")
            if start == -1:
                return None
            end = sql.find("]", start)
            return int(sql[start + 6 : end])
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Protocol: store
    # ------------------------------------------------------------------

    async def store(self, collection_name: str, record: MemoryRecord) -> None:
        """Persist a MemoryRecord into the collection.

        Both the embedding vector and the metadata are written atomically.
        If the record has no embedding the store is skipped (vector search
        would not work).
        """
        try:
            if record.entry.embedding is None:
                logger.warning(
                    "Skipping store for record %s: no embedding", record.record_id
                )
                return

            dim = record.entry.embedding.dimensions
            self._ensure_tables(collection_name, dim)
            safe_name = _sanitize_collection_name(collection_name)
            conn = self._get_connection()

            packed = _pack_vector(record.entry.embedding.values)

            # Insert the vector and retrieve its rowid.
            cursor = conn.execute(
                f"INSERT INTO [{safe_name}_vec](embedding) VALUES (?)",
                (packed,),
            )
            vec_rowid = cursor.lastrowid

            # Insert the metadata row.
            conn.execute(
                f"INSERT OR REPLACE INTO [{safe_name}_meta] "
                "(record_id, vec_rowid, content, memory_type, metadata_json, "
                " tags_json, session_id, created_at, accessed_at, access_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.record_id,
                    vec_rowid,
                    record.entry.content,
                    record.entry.memory_type.value,
                    json.dumps(record.entry.metadata) if record.entry.metadata else None,
                    json.dumps(list(record.entry.tags)) if record.entry.tags else None,
                    record.session_id,
                    record.created_at.isoformat(),
                    record.accessed_at.isoformat(),
                    record.access_count,
                ),
            )

            conn.commit()
        except Exception:  # noqa: BLE001
            logger.error(
                "Failed to store record %s in %s",
                record.record_id,
                collection_name,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Protocol: search
    # ------------------------------------------------------------------

    async def search(
        self,
        collection_names: List[str],
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[MemoryRecord, SimilarityScore]]:
        """Vector similarity search across one or more collections.

        sqlite-vec returns cosine *distance* (0 = identical). We convert to
        similarity as ``max(0, 1 - distance)``.
        """
        results: List[Tuple[MemoryRecord, SimilarityScore]] = []
        packed_query = _pack_vector(query_embedding.values)

        for coll_name in collection_names:
            try:
                safe_name = _sanitize_collection_name(coll_name)
                if safe_name not in self._initialised_collections:
                    # The collection may exist from a prior session.
                    dim = self._get_dimension(coll_name)
                    if dim is None:
                        continue
                    self._ensure_tables(coll_name, dim)

                conn = self._get_connection()

                rows = conn.execute(
                    f"SELECT v.rowid AS vec_rowid, v.distance "
                    f"FROM [{safe_name}_vec] v "
                    f"WHERE v.embedding MATCH ? AND k = ?",
                    (packed_query, top_k),
                ).fetchall()

                for row in rows:
                    vec_rowid = row["vec_rowid"] if isinstance(row, sqlite3.Row) else row[0]
                    distance = row["distance"] if isinstance(row, sqlite3.Row) else row[1]
                    similarity = max(0.0, 1.0 - float(distance))

                    if similarity < min_similarity:
                        continue

                    meta_row = conn.execute(
                        f"SELECT * FROM [{safe_name}_meta] WHERE vec_rowid = ?",
                        (vec_rowid,),
                    ).fetchone()

                    if meta_row is None:
                        continue

                    record = self._row_to_record(meta_row, query_embedding)
                    score = SimilarityScore.cosine(
                        min(1.0, max(0.0, similarity))
                    )
                    results.append((record, score))

            except Exception:  # noqa: BLE001
                logger.error(
                    "Search failed for collection %s", coll_name, exc_info=True
                )

        # Sort by similarity descending and trim to top_k overall.
        results.sort(key=lambda x: x[1].value, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Protocol: get_by_id
    # ------------------------------------------------------------------

    async def get_by_id(
        self, collection_name: str, record_id: str
    ) -> Optional[MemoryRecord]:
        try:
            safe_name = _sanitize_collection_name(collection_name)
            if safe_name not in self._initialised_collections:
                dim = self._get_dimension(collection_name)
                if dim is None:
                    return None
                self._ensure_tables(collection_name, dim)

            conn = self._get_connection()

            row = conn.execute(
                f"SELECT * FROM [{safe_name}_meta] WHERE record_id = ?",
                (record_id,),
            ).fetchone()

            if row is None:
                return None

            return self._row_to_record(row)
        except Exception:  # noqa: BLE001
            logger.error(
                "get_by_id failed for %s/%s",
                collection_name,
                record_id,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Protocol: delete
    # ------------------------------------------------------------------

    async def delete(self, collection_name: str, record_id: str) -> bool:
        try:
            safe_name = _sanitize_collection_name(collection_name)
            if safe_name not in self._initialised_collections:
                dim = self._get_dimension(collection_name)
                if dim is None:
                    return False
                self._ensure_tables(collection_name, dim)

            conn = self._get_connection()

            # Look up the vec_rowid so we can delete from both tables.
            row = conn.execute(
                f"SELECT vec_rowid FROM [{safe_name}_meta] WHERE record_id = ?",
                (record_id,),
            ).fetchone()

            if row is None:
                return False

            vec_rowid = row["vec_rowid"] if isinstance(row, sqlite3.Row) else row[0]

            conn.execute(
                f"DELETE FROM [{safe_name}_vec] WHERE rowid = ?",
                (vec_rowid,),
            )
            conn.execute(
                f"DELETE FROM [{safe_name}_meta] WHERE record_id = ?",
                (record_id,),
            )
            conn.commit()
            return True
        except Exception:  # noqa: BLE001
            logger.error(
                "delete failed for %s/%s",
                collection_name,
                record_id,
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # Protocol: delete_by_age
    # ------------------------------------------------------------------

    async def delete_by_age(
        self, collection_name: str, max_age_days: float
    ) -> int:
        try:
            safe_name = _sanitize_collection_name(collection_name)
            if safe_name not in self._initialised_collections:
                dim = self._get_dimension(collection_name)
                if dim is None:
                    return 0
                self._ensure_tables(collection_name, dim)

            conn = self._get_connection()
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()

            # Find stale rows by accessed_at.
            stale_rows = conn.execute(
                f"SELECT record_id, vec_rowid FROM [{safe_name}_meta] "
                "WHERE accessed_at < ?",
                (cutoff,),
            ).fetchall()

            if not stale_rows:
                return 0

            vec_rowids = []
            record_ids = []
            for row in stale_rows:
                if isinstance(row, sqlite3.Row):
                    record_ids.append(row["record_id"])
                    vec_rowids.append(row["vec_rowid"])
                else:
                    record_ids.append(row[0])
                    vec_rowids.append(row[1])

            # Delete from vec table.
            placeholders = ",".join("?" for _ in vec_rowids)
            conn.execute(
                f"DELETE FROM [{safe_name}_vec] WHERE rowid IN ({placeholders})",
                vec_rowids,
            )

            # Delete from meta table.
            placeholders = ",".join("?" for _ in record_ids)
            conn.execute(
                f"DELETE FROM [{safe_name}_meta] WHERE record_id IN ({placeholders})",
                record_ids,
            )

            conn.commit()
            return len(record_ids)
        except Exception:  # noqa: BLE001
            logger.error(
                "delete_by_age failed for %s (max_age_days=%s)",
                collection_name,
                max_age_days,
                exc_info=True,
            )
            return 0

    # ------------------------------------------------------------------
    # Protocol: ensure_collection
    # ------------------------------------------------------------------

    async def ensure_collection(
        self, collection_name: str, dimension: int
    ) -> None:
        try:
            self._ensure_tables(collection_name, dimension)
        except Exception:  # noqa: BLE001
            logger.error(
                "ensure_collection failed for %s (dim=%d)",
                collection_name,
                dimension,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Protocol: collection_stats
    # ------------------------------------------------------------------

    async def collection_stats(
        self, collection_name: str
    ) -> Optional[Dict[str, Any]]:
        try:
            safe_name = _sanitize_collection_name(collection_name)
            dim = self._get_dimension(collection_name)
            if dim is None:
                return None

            if safe_name not in self._initialised_collections:
                self._ensure_tables(collection_name, dim)

            conn = self._get_connection()

            row = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM [{safe_name}_meta]"
            ).fetchone()
            count = row["cnt"] if isinstance(row, sqlite3.Row) else row[0]

            return {
                "collection_name": collection_name,
                "record_count": count,
                "dimension": dim,
            }
        except Exception:  # noqa: BLE001
            logger.error(
                "collection_stats failed for %s",
                collection_name,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Protocol: count
    # ------------------------------------------------------------------

    async def count(self, collection_name: Optional[str] = None) -> int:
        try:
            conn = self._get_connection()

            if collection_name is not None:
                safe_name = _sanitize_collection_name(collection_name)
                # Check the table exists before querying.
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (f"{safe_name}_meta",),
                ).fetchone()
                if exists is None:
                    return 0
                row = conn.execute(
                    f"SELECT COUNT(*) AS cnt FROM [{safe_name}_meta]"
                ).fetchone()
                return row["cnt"] if isinstance(row, sqlite3.Row) else row[0]

            # Sum across all meta tables.
            tables = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name LIKE '%_meta'"
            ).fetchall()

            total = 0
            for t in tables:
                tname = t["name"] if isinstance(t, sqlite3.Row) else t[0]
                row = conn.execute(
                    f"SELECT COUNT(*) AS cnt FROM [{tname}]"
                ).fetchone()
                total += row["cnt"] if isinstance(row, sqlite3.Row) else row[0]
            return total
        except Exception:  # noqa: BLE001
            logger.error("count failed", exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row | tuple[Any, ...],
        embedding_ref: Optional[EmbeddingVector] = None,
    ) -> MemoryRecord:
        """Reconstruct a MemoryRecord from a metadata row.

        The embedding is *not* stored in the meta table, so the returned
        record's entry will have ``embedding=None`` unless the caller
        supplies ``embedding_ref`` (used only for model_name when we could
        reconstruct it from the vec table, which we do not do for
        performance).
        """
        if isinstance(row, sqlite3.Row):
            record_id: str = row["record_id"]
            content: str = row["content"]
            memory_type_str: str = row["memory_type"]
            metadata_json: Optional[str] = row["metadata_json"]
            tags_json: Optional[str] = row["tags_json"]
            session_id: Optional[str] = row["session_id"]
            created_at_str: str = row["created_at"]
            accessed_at_str: str = row["accessed_at"]
            access_count: int = row["access_count"]
        else:
            # Positional fallback (unlikely with row_factory but safe).
            # Columns: record_id, vec_rowid, content, memory_type,
            #          metadata_json, tags_json, session_id,
            #          created_at, accessed_at, access_count
            record_id = row[0]
            content = row[2]
            memory_type_str = row[3]
            metadata_json = row[4]
            tags_json = row[5]
            session_id = row[6]
            created_at_str = row[7]
            accessed_at_str = row[8]
            access_count = row[9]

        metadata: Dict[str, Any] = (
            json.loads(metadata_json) if metadata_json else {}
        )
        tags: Tuple[str, ...] = (
            tuple(json.loads(tags_json)) if tags_json else ()
        )

        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType(memory_type_str),
            metadata=metadata,
            tags=tags,
            # Embedding not stored in meta — omit.
        )

        return MemoryRecord(
            record_id=record_id,
            entry=entry,
            created_at=datetime.fromisoformat(created_at_str),
            accessed_at=datetime.fromisoformat(accessed_at_str),
            access_count=access_count,
            session_id=session_id,
        )
