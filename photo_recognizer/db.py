from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_EVENT_NAME = "Основное событие"
DEFAULT_EVENT_DATE = ""

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    event_date TEXT NOT NULL DEFAULT '',
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    person_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    top_px INTEGER NOT NULL,
    right_px INTEGER NOT NULL,
    bottom_px INTEGER NOT NULL,
    left_px INTEGER NOT NULL,
    crop_path TEXT,
    context_json TEXT NOT NULL DEFAULT '{}',
    embedding_json TEXT NOT NULL,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id INTEGER NOT NULL,
    engine TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
    UNIQUE(face_id, engine)
);

CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS person_faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    face_id INTEGER NOT NULL,
    confirmed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
    UNIQUE(person_id, face_id),
    UNIQUE(face_id)
);

CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_index ON faces(person_index);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_face_id ON face_embeddings(face_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_engine ON face_embeddings(engine);
CREATE INDEX IF NOT EXISTS idx_person_faces_person_id ON person_faces(person_id);
CREATE INDEX IF NOT EXISTS idx_person_faces_face_id ON person_faces(face_id);
"""


class FaceRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            self._ensure_events_default_column(connection)
            self._ensure_faces_context_column(connection)
            self._ensure_photos_event_column(connection)
            photo_count = int(connection.execute("SELECT COUNT(*) AS value FROM photos").fetchone()["value"])
            needs_backfill = bool(
                connection.execute(
                    "SELECT 1 FROM photos WHERE event_id IS NULL LIMIT 1"
                ).fetchone()
            )
            if photo_count > 0 or needs_backfill:
                self._ensure_default_event(connection)

    def reset(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS person_faces;
                DROP TABLE IF EXISTS persons;
                DROP TABLE IF EXISTS face_embeddings;
                DROP TABLE IF EXISTS faces;
                DROP TABLE IF EXISTS photos;
                DROP TABLE IF EXISTS events;
                """
            )
        self.init_db()

    def get_or_create_default_event(self) -> int:
        with self._connect() as connection:
            return self._ensure_default_event(connection)

    def list_events(self) -> list[sqlite3.Row]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    e.id,
                    e.name,
                    e.event_date,
                    e.is_default,
                    e.created_at,
                    COUNT(DISTINCT p.id) AS photo_count,
                    COUNT(DISTINCT f.id) AS face_count,
                    COUNT(DISTINCT pf.person_id) AS person_count
                FROM events e
                LEFT JOIN photos p ON p.event_id = e.id
                LEFT JOIN faces f ON f.photo_id = p.id
                LEFT JOIN person_faces pf ON pf.face_id = f.id
                GROUP BY e.id
                ORDER BY
                    CASE WHEN e.is_default = 1 THEN 0 ELSE 1 END,
                    e.event_date DESC,
                    e.created_at DESC,
                    e.id DESC
                """
            ).fetchall()
        return list(rows)

    def create_event(self, name: str, event_date: str) -> int:
        normalized_name = (name or "").strip()
        if not normalized_name:
            raise ValueError("Event name is required")

        normalized_date = (event_date or "").strip()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events (name, event_date)
                VALUES (?, ?)
                """,
                (normalized_name, normalized_date),
            )
        return int(cursor.lastrowid)

    def rename_event(self, event_id: int, name: str, event_date: str) -> None:
        normalized_name = (name or "").strip()
        if not normalized_name:
            raise ValueError("Event name is required")

        normalized_date = (event_date or "").strip()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE events
                SET name = ?, event_date = ?
                WHERE id = ?
                """,
                (normalized_name, normalized_date, event_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("Event not found")

    def delete_event(self, event_id: int) -> None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    e.id,
                    e.is_default,
                    COUNT(p.id) AS photo_count
                FROM events e
                LEFT JOIN photos p ON p.event_id = e.id
                WHERE e.id = ?
                GROUP BY e.id
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Event not found")
            if int(row["is_default"]) == 1:
                raise ValueError("Default event cannot be deleted")
            if int(row["photo_count"]) > 0:
                raise ValueError("Only empty events can be deleted")

            connection.execute("DELETE FROM events WHERE id = ?", (event_id,))

    def get_event(self, event_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    e.id,
                    e.name,
                    e.event_date,
                    e.is_default,
                    e.created_at,
                    COUNT(DISTINCT p.id) AS photo_count,
                    COUNT(DISTINCT f.id) AS face_count,
                    COUNT(DISTINCT pf.person_id) AS person_count
                FROM events e
                LEFT JOIN photos p ON p.event_id = e.id
                LEFT JOIN faces f ON f.photo_id = p.id
                LEFT JOIN person_faces pf ON pf.face_id = f.id
                WHERE e.id = ?
                GROUP BY e.id
                """,
                (event_id,),
            ).fetchone()
        return row

    def upsert_photo(self, relative_path: str, width: int, height: int, event_id: int) -> int:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO photos (event_id, relative_path, width, height)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(relative_path)
                DO UPDATE SET
                    event_id = excluded.event_id,
                    width = excluded.width,
                    height = excluded.height,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (event_id, relative_path, width, height),
            )
            row = connection.execute(
                "SELECT id FROM photos WHERE relative_path = ?",
                (relative_path,),
            ).fetchone()
        return int(row["id"])

    def replace_faces(self, photo_id: int, faces: list[dict[str, Any]]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))

            for face in faces:
                embeddings = {
                    engine: embedding
                    for engine, embedding in face["embeddings"].items()
                    if embedding is not None
                }
                embedding_versions = face.get("embedding_versions", {})
                primary_embedding = next(iter(embeddings.values()), [])
                cursor = connection.execute(
                    """
                    INSERT INTO faces (
                        photo_id,
                        person_index,
                        label,
                        top_px,
                        right_px,
                        bottom_px,
                        left_px,
                        crop_path,
                        context_json,
                        embedding_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        photo_id,
                        face["person_index"],
                        face["label"],
                        face["top_px"],
                        face["right_px"],
                        face["bottom_px"],
                        face["left_px"],
                        face["crop_path"],
                        json.dumps(face.get("context", {})),
                        json.dumps(primary_embedding),
                    ),
                )
                face_id = int(cursor.lastrowid)
                for engine, embedding in embeddings.items():
                    connection.execute(
                        """
                        INSERT INTO face_embeddings (
                            face_id,
                            engine,
                            embedding_json,
                            embedding_version
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            face_id,
                            engine,
                            json.dumps(embedding),
                            str(embedding_versions.get(engine, "1")),
                        ),
                    )

    def list_photos(
        self,
        limit: int = 60,
        offset: int = 0,
        event_id: int | None = None,
    ) -> list[sqlite3.Row]:
        where_clause = "WHERE p.event_id = ?" if event_id is not None else ""
        params: list[Any] = [event_id] if event_id is not None else []
        params.extend([limit, offset])

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    p.id,
                    p.event_id,
                    p.relative_path,
                    p.width,
                    p.height,
                    e.name AS event_name,
                    e.event_date,
                    COUNT(f.id) AS face_count
                FROM photos p
                JOIN events e ON e.id = p.event_id
                LEFT JOIN faces f ON f.photo_id = p.id
                {where_clause}
                GROUP BY p.id
                ORDER BY p.relative_path
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return list(rows)

    def get_photo(self, photo_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    p.id,
                    p.event_id,
                    p.relative_path,
                    p.width,
                    p.height,
                    e.name AS event_name,
                    e.event_date,
                    COUNT(f.id) AS face_count
                FROM photos p
                JOIN events e ON e.id = p.event_id
                LEFT JOIN faces f ON f.photo_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (photo_id,),
            ).fetchone()
        return row

    def get_photo_faces(self, photo_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    f.id,
                    f.photo_id,
                    f.person_index,
                    f.label,
                    f.top_px,
                    f.right_px,
                    f.bottom_px,
                    f.left_px,
                    f.crop_path,
                    p.id AS person_id,
                    p.display_name AS person_name
                FROM faces f
                LEFT JOIN person_faces pf ON pf.face_id = f.id
                LEFT JOIN persons p ON p.id = pf.person_id
                WHERE f.photo_id = ?
                ORDER BY f.person_index
                """,
                (photo_id,),
            ).fetchall()
        return list(rows)

    def get_face(self, face_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    f.id,
                    f.photo_id,
                    f.person_index,
                    f.label,
                    f.top_px,
                    f.right_px,
                    f.bottom_px,
                    f.left_px,
                    f.crop_path,
                    f.context_json,
                    p.relative_path,
                    p.width,
                    p.height,
                    p.event_id,
                    e.name AS event_name,
                    e.event_date,
                    person.id AS person_id,
                    person.display_name AS person_name
                FROM faces f
                JOIN photos p ON p.id = f.photo_id
                JOIN events e ON e.id = p.event_id
                LEFT JOIN person_faces pf ON pf.face_id = f.id
                LEFT JOIN persons person ON person.id = pf.person_id
                WHERE f.id = ?
                """,
                (face_id,),
            ).fetchone()
        return row

    def get_face_embeddings(self, face_id: int) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    engine,
                    embedding_json,
                    embedding_version
                FROM face_embeddings
                WHERE face_id = ?
                """,
                (face_id,),
            ).fetchall()

        return {
            str(row["engine"]): {
                "embedding": json.loads(row["embedding_json"]),
                "version": str(row["embedding_version"]),
            }
            for row in rows
        }

    def list_indexed_faces_for_engine(
        self,
        engine: str,
        event_id: int | None = None,
    ) -> list[sqlite3.Row]:
        where_clause = "WHERE p.event_id = ?" if event_id is not None else ""
        params: list[Any] = [engine]
        if event_id is not None:
            params.append(event_id)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    f.id,
                    f.photo_id,
                    f.person_index,
                    f.label,
                    f.top_px,
                    f.right_px,
                    f.bottom_px,
                    f.left_px,
                    f.crop_path,
                    f.context_json,
                    p.relative_path,
                    p.width,
                    p.height,
                    p.event_id,
                    e.name AS event_name,
                    e.event_date,
                    fe.embedding_json,
                    fe.embedding_version,
                    person.id AS person_id,
                    person.display_name AS person_name
                FROM faces f
                JOIN photos p ON p.id = f.photo_id
                JOIN events e ON e.id = p.event_id
                JOIN face_embeddings fe
                    ON fe.face_id = f.id
                    AND fe.engine = ?
                LEFT JOIN person_faces pf ON pf.face_id = f.id
                LEFT JOIN persons person ON person.id = pf.person_id
                {where_clause}
                ORDER BY p.relative_path, f.person_index
                """,
                params,
            ).fetchall()
        return list(rows)

    def get_embeddings_for_faces(self, face_ids: list[int], engine: str) -> dict[int, dict[str, Any]]:
        if not face_ids:
            return {}

        placeholders = ", ".join("?" for _ in face_ids)
        query = f"""
            SELECT
                face_id,
                embedding_json,
                embedding_version
            FROM face_embeddings
            WHERE engine = ?
              AND face_id IN ({placeholders})
        """
        params: list[Any] = [engine, *face_ids]

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        return {
            int(row["face_id"]): {
                "embedding": json.loads(row["embedding_json"]),
                "version": str(row["embedding_version"]),
            }
            for row in rows
        }

    def create_person(self, display_name: str, face_id: int) -> int:
        normalized_name = (display_name or "").strip()
        if not normalized_name:
            raise ValueError("Person name is required")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO persons (display_name)
                VALUES (?)
                """,
                (normalized_name,),
            )
            person_id = int(cursor.lastrowid)
            self._assign_face_to_person(connection, person_id, face_id)
        return person_id

    def rename_person(self, person_id: int, display_name: str) -> None:
        normalized_name = (display_name or "").strip()
        if not normalized_name:
            raise ValueError("Person name is required")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE persons
                SET display_name = ?
                WHERE id = ?
                """,
                (normalized_name, person_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("Person not found")

    def delete_person(self, person_id: int) -> None:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            if cursor.rowcount == 0:
                raise ValueError("Person not found")

    def attach_face_to_person(self, person_id: int, face_id: int) -> None:
        with self._connect() as connection:
            self._assign_face_to_person(connection, person_id, face_id)

    def list_persons(self, event_id: int | None = None) -> list[sqlite3.Row]:
        params: list[Any] = []
        if event_id is None:
            where_clause = ""
        else:
            where_clause = "WHERE photo.event_id = ?"
            params.append(event_id)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    person.id,
                    person.display_name,
                    person.created_at,
                    COUNT(DISTINCT pf.face_id) AS confirmed_face_count,
                    COUNT(DISTINCT face.photo_id) AS photo_count
                FROM persons person
                LEFT JOIN person_faces pf ON pf.person_id = person.id
                LEFT JOIN faces face ON face.id = pf.face_id
                LEFT JOIN photos photo ON photo.id = face.photo_id
                {where_clause}
                GROUP BY person.id
                HAVING confirmed_face_count > 0
                ORDER BY photo_count DESC, person.display_name COLLATE NOCASE
                """,
                params,
            ).fetchall()
        return list(rows)

    def get_person(self, person_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    person.id,
                    person.display_name,
                    person.created_at,
                    COUNT(DISTINCT pf.face_id) AS confirmed_face_count,
                    COUNT(DISTINCT face.photo_id) AS photo_count
                FROM persons person
                LEFT JOIN person_faces pf ON pf.person_id = person.id
                LEFT JOIN faces face ON face.id = pf.face_id
                WHERE person.id = ?
                GROUP BY person.id
                """,
                (person_id,),
            ).fetchone()
        return row

    def get_person_faces(self, person_id: int, event_id: int | None = None) -> list[sqlite3.Row]:
        params: list[Any] = [person_id]
        where_clause = ""
        if event_id is not None:
            where_clause = "AND photo.event_id = ?"
            params.append(event_id)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    face.id,
                    face.photo_id,
                    face.person_index,
                    face.top_px,
                    face.right_px,
                    face.bottom_px,
                    face.left_px,
                    face.crop_path,
                    face.context_json,
                    photo.relative_path,
                    photo.width,
                    photo.height,
                    photo.event_id,
                    event.name AS event_name,
                    event.event_date
                FROM person_faces pf
                JOIN faces face ON face.id = pf.face_id
                JOIN photos photo ON photo.id = face.photo_id
                JOIN events event ON event.id = photo.event_id
                WHERE pf.person_id = ?
                {where_clause}
                ORDER BY pf.confirmed_at ASC, face.id ASC
                """,
                params,
            ).fetchall()
        return list(rows)

    def get_person_profile_embeddings(self, person_id: int) -> dict[str, list[list[float]]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    fe.engine,
                    fe.embedding_json
                FROM person_faces pf
                JOIN face_embeddings fe ON fe.face_id = pf.face_id
                WHERE pf.person_id = ?
                ORDER BY fe.engine, fe.face_id
                """,
                (person_id,),
            ).fetchall()

        grouped: dict[str, list[list[float]]] = {}
        for row in rows:
            grouped.setdefault(str(row["engine"]), []).append(json.loads(row["embedding_json"]))
        return grouped

    def get_person_contexts(self, person_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    face.context_json
                FROM person_faces pf
                JOIN faces face ON face.id = pf.face_id
                WHERE pf.person_id = ?
                ORDER BY pf.confirmed_at ASC
                """,
                (person_id,),
            ).fetchall()
        return [json.loads(row["context_json"] or "{}") for row in rows]

    def get_stats(self, event_id: int | None = None) -> dict[str, int]:
        params: list[Any] = []
        if event_id is None:
            photo_where = ""
            face_where = ""
            person_where = ""
        else:
            photo_where = "WHERE event_id = ?"
            face_where = """
                WHERE photo_id IN (
                    SELECT id FROM photos WHERE event_id = ?
                )
            """
            person_where = """
                WHERE pf.face_id IN (
                    SELECT face.id
                    FROM faces face
                    JOIN photos photo ON photo.id = face.photo_id
                    WHERE photo.event_id = ?
                )
            """
            params.append(event_id)

        with self._connect() as connection:
            photo_count = connection.execute(
                f"SELECT COUNT(*) AS value FROM photos {photo_where}",
                params,
            ).fetchone()["value"]
            face_count = connection.execute(
                f"SELECT COUNT(*) AS value FROM faces {face_where}",
                params,
            ).fetchone()["value"]
            person_count = connection.execute(
                f"SELECT COUNT(DISTINCT pf.person_id) AS value FROM person_faces pf {person_where}",
                params,
            ).fetchone()["value"]
        return {
            "photo_count": int(photo_count),
            "face_count": int(face_count),
            "person_count": int(person_count),
        }

    def _assign_face_to_person(
        self,
        connection: sqlite3.Connection,
        person_id: int,
        face_id: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO person_faces (person_id, face_id)
            VALUES (?, ?)
            ON CONFLICT(face_id)
            DO UPDATE SET
                person_id = excluded.person_id,
                confirmed_at = CURRENT_TIMESTAMP
            """,
            (person_id, face_id),
        )

    def _ensure_faces_context_column(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(faces)").fetchall()
        }
        if "context_json" not in columns:
            connection.execute(
                "ALTER TABLE faces ADD COLUMN context_json TEXT NOT NULL DEFAULT '{}'"
            )

    def _ensure_photos_event_column(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(photos)").fetchall()
        }
        if "event_id" not in columns:
            connection.execute("ALTER TABLE photos ADD COLUMN event_id INTEGER")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_photos_event_id ON photos(event_id)")

    def _ensure_events_default_column(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(events)").fetchall()
        }
        if "is_default" not in columns:
            connection.execute("ALTER TABLE events ADD COLUMN is_default INTEGER NOT NULL DEFAULT 0")

        flagged_row = connection.execute(
            "SELECT id FROM events WHERE is_default = 1 ORDER BY id LIMIT 1"
        ).fetchone()
        if flagged_row is not None:
            connection.execute(
                "UPDATE events SET is_default = CASE WHEN id = ? THEN 1 ELSE 0 END",
                (int(flagged_row["id"]),),
            )
            return

        legacy_default = connection.execute(
            """
            SELECT id
            FROM events
            WHERE name = ? AND event_date = ?
            ORDER BY id
            LIMIT 1
            """,
            (DEFAULT_EVENT_NAME, DEFAULT_EVENT_DATE),
        ).fetchone()
        if legacy_default is not None:
            connection.execute(
                "UPDATE events SET is_default = CASE WHEN id = ? THEN 1 ELSE 0 END",
                (int(legacy_default["id"]),),
            )

    def _ensure_default_event(self, connection: sqlite3.Connection) -> int:
        row = connection.execute(
            """
            SELECT id
            FROM events
            WHERE is_default = 1
            ORDER BY id
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            cursor = connection.execute(
                """
                INSERT INTO events (name, event_date, is_default)
                VALUES (?, ?, 1)
                """,
                (DEFAULT_EVENT_NAME, DEFAULT_EVENT_DATE),
            )
            default_event_id = int(cursor.lastrowid)
        else:
            default_event_id = int(row["id"])

        connection.execute(
            "UPDATE events SET is_default = CASE WHEN id = ? THEN 1 ELSE 0 END",
            (default_event_id,),
        )

        connection.execute(
            "UPDATE photos SET event_id = ? WHERE event_id IS NULL",
            (default_event_id,),
        )
        return default_event_id

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
