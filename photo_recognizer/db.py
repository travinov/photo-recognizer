from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relative_path TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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

CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_index ON faces(person_index);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_face_id ON face_embeddings(face_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_engine ON face_embeddings(engine);
"""


class FaceRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def reset(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS face_embeddings;
                DROP TABLE IF EXISTS faces;
                DROP TABLE IF EXISTS photos;
                """
            )
        self.init_db()

    def upsert_photo(self, relative_path: str, width: int, height: int) -> int:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO photos (relative_path, width, height)
                VALUES (?, ?, ?)
                ON CONFLICT(relative_path)
                DO UPDATE SET
                    width = excluded.width,
                    height = excluded.height,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (relative_path, width, height),
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
                        embedding_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def list_photos(self, limit: int = 60, offset: int = 0) -> list[sqlite3.Row]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    p.id,
                    p.relative_path,
                    p.width,
                    p.height,
                    COUNT(f.id) AS face_count
                FROM photos p
                LEFT JOIN faces f ON f.photo_id = p.id
                GROUP BY p.id
                ORDER BY p.relative_path
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return list(rows)

    def get_photo(self, photo_id: int) -> sqlite3.Row | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    p.id,
                    p.relative_path,
                    p.width,
                    p.height,
                    COUNT(f.id) AS face_count
                FROM photos p
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
                    id,
                    photo_id,
                    person_index,
                    label,
                    top_px,
                    right_px,
                    bottom_px,
                    left_px,
                    crop_path
                FROM faces
                WHERE photo_id = ?
                ORDER BY person_index
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
                    p.relative_path,
                    p.width,
                    p.height
                FROM faces f
                JOIN photos p ON p.id = f.photo_id
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

    def list_indexed_faces_for_engine(self, engine: str) -> list[sqlite3.Row]:
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
                    p.relative_path,
                    p.width,
                    p.height,
                    fe.embedding_json,
                    fe.embedding_version
                FROM faces f
                JOIN photos p ON p.id = f.photo_id
                JOIN face_embeddings fe
                    ON fe.face_id = f.id
                    AND fe.engine = ?
                ORDER BY p.relative_path, f.person_index
                """,
                (engine,),
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

    def get_stats(self) -> dict[str, int]:
        with self._connect() as connection:
            photo_count = connection.execute("SELECT COUNT(*) AS value FROM photos").fetchone()["value"]
            face_count = connection.execute("SELECT COUNT(*) AS value FROM faces").fetchone()["value"]
        return {"photo_count": int(photo_count), "face_count": int(face_count)}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
