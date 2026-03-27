from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from photo_recognizer.config import Settings
from photo_recognizer.db import FaceRepository
from photo_recognizer.engines import FaceEngine, build_engine
from photo_recognizer.face_engine import iter_image_files, load_image_array, save_face_crop
from photo_recognizer.models import DetectedFace


class IndexService:
    def __init__(self, settings: Settings, repository: FaceRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.primary_engine = build_engine(settings.primary_engine, settings)
        self.verify_engine = build_engine(settings.verify_engine, settings)

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
        width, height, faces = detect_faces_for_path(
            image_path=image_path,
            max_image_size=self.settings.max_image_size,
            primary_engine=self.primary_engine,
            verify_engine=self.verify_engine,
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
                    "embeddings": face.embeddings,
                    "embedding_versions": {
                        self.primary_engine.name: self.primary_engine.embedding_version,
                        self.verify_engine.name: self.verify_engine.embedding_version,
                    },
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
        self.primary_engine = build_engine(settings.primary_engine, settings)
        self.verify_engine = build_engine(settings.verify_engine, settings)

    def store_query_file(self, filename: str, content: bytes) -> Path:
        suffix = Path(filename or "query.jpg").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"

        output_path = self.settings.query_upload_dir / f"{uuid4().hex}{suffix}"
        output_path.write_bytes(content)
        return output_path

    def is_supported_image(self, filename: str | None) -> bool:
        suffix = Path(filename or "").suffix.lower()
        return suffix in {".jpg", ".jpeg", ".png", ".webp"}

    def search(self, query_path: Path) -> dict[str, object]:
        query_width, query_height, query_faces = detect_faces_for_path(
            image_path=query_path,
            max_image_size=self.settings.max_image_size,
            primary_engine=self.primary_engine,
            verify_engine=self.verify_engine,
        )

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
            matches = self._find_matches(face.embeddings)
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
            "primary_engine": self.primary_engine.name,
            "verify_engine": self.verify_engine.name,
            "primary_engine_label": build_engine_label(self.primary_engine.name),
            "verify_engine_label": build_engine_label(self.verify_engine.name),
            "primary_threshold": self.settings.insightface_threshold,
            "verify_threshold": self.settings.dlib_threshold,
        }

    def search_many(self, uploaded_files: list[tuple[str, bytes]]) -> dict[str, object]:
        photo_results: list[dict[str, object]] = []

        for filename, content in uploaded_files:
            if not content or not self.is_supported_image(filename):
                continue

            stored_path = self.store_query_file(filename, content)
            result = self.search(stored_path)
            photo_results.append(
                {
                    "filename": filename,
                    "query_path": result["query_path"],
                    "query_width": result["query_width"],
                    "query_height": result["query_height"],
                    "query_faces": result["query_faces"],
                    "primary_engine_label": result["primary_engine_label"],
                    "verify_engine_label": result["verify_engine_label"],
                    "primary_threshold": result["primary_threshold"],
                    "verify_threshold": result["verify_threshold"],
                }
            )

        return {
            "photos": photo_results,
            "photo_count": len(photo_results),
        }

    def search_indexed_face(self, face_id: int) -> dict[str, object] | None:
        query_face = self.repository.get_face(face_id)
        if query_face is None:
            return None

        query_embeddings = {
            engine: payload["embedding"]
            for engine, payload in self.repository.get_face_embeddings(face_id).items()
        }
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
                query_embeddings,
                exclude_face_id=int(query_face["id"]),
            ),
        }

        return {
            "query_path": query_face["relative_path"],
            "query_width": int(query_face["width"]),
            "query_height": int(query_face["height"]),
            "query_faces": [query_face_result],
            "primary_engine": self.primary_engine.name,
            "verify_engine": self.verify_engine.name,
            "primary_engine_label": build_engine_label(self.primary_engine.name),
            "verify_engine_label": build_engine_label(self.verify_engine.name),
            "primary_threshold": self.settings.insightface_threshold,
            "verify_threshold": self.settings.dlib_threshold,
            "source_photo_id": int(query_face["photo_id"]),
            "source_relative_path": query_face["relative_path"],
        }

    def _find_matches(
        self,
        query_embeddings: dict[str, list[float]],
        exclude_face_id: int | None = None,
    ) -> list[dict[str, object]]:
        primary_embedding = query_embeddings.get(self.primary_engine.name)
        verify_embedding = query_embeddings.get(self.verify_engine.name)
        if primary_embedding is None or verify_embedding is None:
            return []

        indexed_faces = self.repository.list_indexed_faces_for_engine(self.primary_engine.name)
        if not indexed_faces:
            return []

        primary_vectors = np.array(
            [json.loads(row["embedding_json"]) for row in indexed_faces],
            dtype=np.float32,
        )
        primary_scores = cosine_similarity(primary_vectors, np.array(primary_embedding, dtype=np.float32))
        ranked = sorted(enumerate(primary_scores), key=lambda item: float(item[1]), reverse=True)

        candidate_rows: list[tuple[object, float]] = []
        for row_index, primary_score in ranked:
            row = indexed_faces[row_index]
            if exclude_face_id is not None and int(row["id"]) == exclude_face_id:
                continue
            if float(primary_score) < self.settings.insightface_threshold:
                break
            candidate_rows.append((row, float(primary_score)))
            if len(candidate_rows) >= self.settings.candidate_top_n:
                break

        if not candidate_rows:
            return []

        candidate_ids = [int(row["id"]) for row, _ in candidate_rows]
        verify_embeddings = self.repository.get_embeddings_for_faces(candidate_ids, self.verify_engine.name)

        matches_by_photo: dict[int, dict[str, object]] = {}
        for row, primary_score in candidate_rows:
            face_id = int(row["id"])
            verify_payload = verify_embeddings.get(face_id)
            if verify_payload is None:
                continue

            verify_score = euclidean_distance(
                np.array(verify_embedding, dtype=np.float32),
                np.array(verify_payload["embedding"], dtype=np.float32),
            )
            confidence = classify_match(
                primary_score=primary_score,
                verify_score=verify_score,
                primary_threshold=self.settings.insightface_threshold,
                verify_threshold=self.settings.dlib_threshold,
            )
            if confidence == "rejected":
                continue

            photo_id = int(row["photo_id"])
            match = {
                "photo_id": photo_id,
                "relative_path": row["relative_path"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "face_id": face_id,
                "person_index": int(row["person_index"]),
                "label": build_indexed_face_label(face_id),
                "top_px": int(row["top_px"]),
                "right_px": int(row["right_px"]),
                "bottom_px": int(row["bottom_px"]),
                "left_px": int(row["left_px"]),
                "crop_path": row["crop_path"],
                "confidence": confidence,
                "confidence_label": build_confidence_label(confidence),
                "primary_score": float(primary_score),
                "verify_score": float(verify_score),
            }

            existing = matches_by_photo.get(photo_id)
            if existing is None or match_sort_key(match) < match_sort_key(existing):
                matches_by_photo[photo_id] = match

        matches = sorted(matches_by_photo.values(), key=match_sort_key)
        return matches[: self.settings.top_k_matches]


def detect_faces_for_path(
    image_path: Path,
    max_image_size: int,
    primary_engine: FaceEngine,
    verify_engine: FaceEngine,
) -> tuple[int, int, list[DetectedFace]]:
    image_array, scale, original_width, original_height = load_image_array(image_path, max_image_size)
    faces = primary_engine.detect_faces(
        image_array=image_array,
        scale=scale,
        original_width=original_width,
        original_height=original_height,
    )

    if faces and verify_engine.name != primary_engine.name:
        verify_embeddings = verify_engine.embed_faces(
            image_array=image_array,
            scale=scale,
            original_width=original_width,
            original_height=original_height,
            faces=faces,
        )
        for face, embedding in zip(faces, verify_embeddings):
            if embedding is not None:
                face.embeddings[verify_engine.name] = embedding

    return original_width, original_height, faces


def cosine_similarity(embeddings: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.empty((0,), dtype=np.float32)

    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0:
        return np.zeros((len(embeddings),), dtype=np.float32)

    matrix_norms = np.linalg.norm(embeddings, axis=1)
    safe_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
    normalized_matrix = embeddings / safe_norms[:, None]
    normalized_query = query_vector / query_norm
    return normalized_matrix @ normalized_query


def euclidean_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def classify_match(
    primary_score: float,
    verify_score: float,
    primary_threshold: float,
    verify_threshold: float,
) -> str:
    strong_primary_threshold = min(0.99, primary_threshold + 0.1)
    relaxed_verify_threshold = verify_threshold + 0.05

    if primary_score >= strong_primary_threshold and verify_score <= verify_threshold:
        return "high"
    if primary_score >= primary_threshold and verify_score <= relaxed_verify_threshold:
        return "medium"
    return "rejected"


def match_sort_key(match: dict[str, object]) -> tuple[float, float, float]:
    confidence_rank = 0 if match["confidence"] == "high" else 1
    return (
        float(confidence_rank),
        -float(match["primary_score"]),
        float(match["verify_score"]),
    )


def build_crop_filename(base_name: str, person_index: int) -> str:
    sanitized = base_name.replace("/", "_").replace(" ", "_")
    return f"{sanitized}_person_{person_index}.jpg"


def build_indexed_face_label(face_id: int) -> str:
    return f"Лицо {face_id}"


def build_engine_label(engine: str) -> str:
    return "InsightFace" if engine == "insightface" else "dlib"


def build_confidence_label(confidence: str) -> str:
    return "Высокая" if confidence == "high" else "Средняя"
