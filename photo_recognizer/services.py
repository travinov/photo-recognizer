from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from photo_recognizer.config import Settings
from photo_recognizer.db import FaceRepository
from photo_recognizer.face_engine import detect_faces, iter_image_files, save_face_crop


class IndexService:
    def __init__(self, settings: Settings, repository: FaceRepository) -> None:
        self.settings = settings
        self.repository = repository

    def rebuild(self, reset: bool = True) -> dict[str, int]:
        self.repository.init_db()
        if reset:
            self.repository.reset()
            self._reset_face_crops()

        photo_count = 0
        face_count = 0

        for image_path in iter_image_files(self.settings.images_dir):
            indexed_faces = self.index_single_photo(image_path)
            photo_count += 1
            face_count += indexed_faces

        return {"photos_indexed": photo_count, "faces_indexed": face_count}

    def index_single_photo(self, image_path: Path) -> int:
        relative_path = image_path.relative_to(self.settings.images_dir).as_posix()
        width, height, faces = detect_faces(
            image_path=image_path,
            detection_model=self.settings.detection_model,
            max_image_size=self.settings.max_image_size,
        )

        photo_id = self.repository.upsert_photo(relative_path=relative_path, width=width, height=height)
        stored_faces: list[dict[str, object]] = []

        for face in faces:
            crop_filename = build_crop_filename(relative_path, face.person_index)
            crop_path = self.settings.face_crop_dir / crop_filename
            save_face_crop(
                image_path=image_path,
                output_path=crop_path,
                top=face.top,
                right=face.right,
                bottom=face.bottom,
                left=face.left,
            )
            stored_faces.append(
                {
                    "person_index": face.person_index,
                    "label": face.label,
                    "top_px": face.top,
                    "right_px": face.right,
                    "bottom_px": face.bottom,
                    "left_px": face.left,
                    "crop_path": crop_filename,
                    "embedding": face.embedding,
                }
            )

        self.repository.replace_faces(photo_id=photo_id, faces=stored_faces)
        return len(stored_faces)

    def _reset_face_crops(self) -> None:
        if self.settings.face_crop_dir.exists():
            shutil.rmtree(self.settings.face_crop_dir)
        self.settings.face_crop_dir.mkdir(parents=True, exist_ok=True)


class SearchService:
    def __init__(self, settings: Settings, repository: FaceRepository) -> None:
        self.settings = settings
        self.repository = repository

    def store_query_file(self, filename: str, content: bytes) -> Path:
        suffix = Path(filename or "query.jpg").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"

        output_path = self.settings.query_upload_dir / f"{uuid4().hex}{suffix}"
        output_path.write_bytes(content)
        return output_path

    def search(self, query_path: Path) -> dict[str, object]:
        query_width, query_height, query_faces = detect_faces(
            image_path=query_path,
            detection_model=self.settings.detection_model,
            max_image_size=self.settings.max_image_size,
        )
        indexed_faces = self.repository.list_indexed_faces()

        if indexed_faces:
            embeddings = np.array(
                [json.loads(row["embedding_json"]) for row in indexed_faces],
                dtype=np.float32,
            )
        else:
            embeddings = np.empty((0, 128), dtype=np.float32)

        query_results: list[dict[str, object]] = []
        for face in query_faces:
            crop_filename = build_crop_filename(query_path.name, face.person_index)
            crop_path = self.settings.face_crop_dir / f"query_{crop_filename}"
            save_face_crop(
                image_path=query_path,
                output_path=crop_path,
                top=face.top,
                right=face.right,
                bottom=face.bottom,
                left=face.left,
            )
            matches = self._find_matches(face.embedding, indexed_faces, embeddings)
            query_results.append(
                {
                    "person_index": face.person_index,
                    "label": f"Запрос {face.person_index}",
                    "top_px": face.top,
                    "right_px": face.right,
                    "bottom_px": face.bottom,
                    "left_px": face.left,
                    "crop_path": crop_path.relative_to(self.settings.storage_dir).as_posix(),
                    "matches": matches,
                }
            )

        return {
            "query_path": query_path.relative_to(self.settings.storage_dir).as_posix(),
            "query_width": query_width,
            "query_height": query_height,
            "query_faces": query_results,
            "threshold": self.settings.match_threshold,
        }

    def search_indexed_face(self, face_id: int) -> dict[str, object] | None:
        query_face = self.repository.get_face(face_id)
        if query_face is None:
            return None

        indexed_faces = self.repository.list_indexed_faces()
        if indexed_faces:
            embeddings = np.array(
                [json.loads(row["embedding_json"]) for row in indexed_faces],
                dtype=np.float32,
            )
        else:
            embeddings = np.empty((0, 128), dtype=np.float32)

        query_face_result = {
            "face_id": int(query_face["id"]),
            "person_index": int(query_face["person_index"]),
            "label": build_indexed_face_label(int(query_face["id"])),
            "top_px": int(query_face["top_px"]),
            "right_px": int(query_face["right_px"]),
            "bottom_px": int(query_face["bottom_px"]),
            "left_px": int(query_face["left_px"]),
            "crop_path": query_face["crop_path"],
            "matches": self._find_matches(
                json.loads(query_face["embedding_json"]),
                indexed_faces,
                embeddings,
                exclude_face_id=int(query_face["id"]),
            ),
        }

        return {
            "query_path": query_face["relative_path"],
            "query_width": int(query_face["width"]),
            "query_height": int(query_face["height"]),
            "query_faces": [query_face_result],
            "threshold": self.settings.match_threshold,
            "source_photo_id": int(query_face["photo_id"]),
            "source_relative_path": query_face["relative_path"],
        }

    def _find_matches(
        self,
        query_embedding: list[float],
        indexed_faces: list[object],
        embeddings: np.ndarray,
        exclude_face_id: int | None = None,
    ) -> list[dict[str, object]]:
        if len(indexed_faces) == 0:
            return []

        query_vector = np.array(query_embedding, dtype=np.float32)
        distances = np.linalg.norm(embeddings - query_vector, axis=1)
        ranked = sorted(enumerate(distances), key=lambda item: float(item[1]))

        matches_by_photo: dict[int, dict[str, object]] = {}
        for row_index, distance in ranked:
            row = indexed_faces[row_index]
            if exclude_face_id is not None and int(row["id"]) == exclude_face_id:
                continue

            if float(distance) > self.settings.match_threshold:
                continue

            photo_id = int(row["photo_id"])
            existing = matches_by_photo.get(photo_id)
            if existing is not None and existing["distance"] <= float(distance):
                continue

            matches_by_photo[photo_id] = {
                "photo_id": photo_id,
                "relative_path": row["relative_path"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "face_id": int(row["id"]),
                "person_index": int(row["person_index"]),
                "label": build_indexed_face_label(int(row["id"])),
                "top_px": int(row["top_px"]),
                "right_px": int(row["right_px"]),
                "bottom_px": int(row["bottom_px"]),
                "left_px": int(row["left_px"]),
                "crop_path": row["crop_path"],
                "distance": float(distance),
            }

        matches = sorted(matches_by_photo.values(), key=lambda item: item["distance"])
        return matches[: self.settings.top_k_matches]


def build_crop_filename(base_name: str, person_index: int) -> str:
    sanitized = base_name.replace("/", "_").replace(" ", "_")
    return f"{sanitized}_person_{person_index}.jpg"


def build_indexed_face_label(face_id: int) -> str:
    return f"Лицо {face_id}"
