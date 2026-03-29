from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np

from photo_recognizer.config import Settings
from photo_recognizer.db import FaceRepository
from photo_recognizer.engines import FaceEngine, build_engine
from photo_recognizer.face_engine import iter_image_files, load_image_array, load_original_image_array, save_face_crop
from photo_recognizer.face_features import build_face_context_features, compute_context_score, enrich_face_embeddings
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

        default_event_id = self.repository.get_or_create_default_event()
        photo_count = 0
        face_count = 0

        for image_path in iter_image_files(self.settings.images_dir):
            if self._is_runtime_import_path(image_path):
                continue
            indexed_faces = self.index_single_photo(image_path=image_path, event_id=default_event_id)
            photo_count += 1
            face_count += indexed_faces

        return {"photos_indexed": photo_count, "faces_indexed": face_count}

    def index_single_photo(self, image_path: Path, event_id: int) -> int:
        relative_path = image_path.relative_to(self.settings.images_dir).as_posix()
        width, height, faces = detect_faces_for_path(
            image_path=image_path,
            max_image_size=self.settings.max_image_size,
            primary_engine=self.primary_engine,
            verify_engine=self.verify_engine,
            small_face_threshold=self.settings.small_face_threshold,
        )

        photo_id = self.repository.upsert_photo(
            relative_path=relative_path,
            width=width,
            height=height,
            event_id=event_id,
        )
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
                    "context": face.context_features,
                    "embeddings": face.embeddings,
                    "embedding_versions": {
                        self.primary_engine.name: self.primary_engine.embedding_version,
                        self.verify_engine.name: self.verify_engine.embedding_version,
                    },
                }
            )

        self.repository.replace_faces(photo_id=photo_id, faces=stored_faces)
        return len(stored_faces)

    def import_uploaded_files(
        self,
        event_id: int,
        uploaded_files: list[tuple[str, bytes]],
    ) -> dict[str, int]:
        event_dir = self.settings.images_dir / "_events" / f"event_{event_id}"
        event_dir.mkdir(parents=True, exist_ok=True)

        photo_count = 0
        face_count = 0
        for filename, content in uploaded_files:
            if not content or not is_supported_image_name(filename):
                continue

            output_path = self._store_event_photo(event_dir, filename, content)
            indexed_faces = self.index_single_photo(image_path=output_path, event_id=event_id)
            photo_count += 1
            face_count += indexed_faces

        return {"photos_indexed": photo_count, "faces_indexed": face_count}

    def _store_event_photo(self, event_dir: Path, filename: str, content: bytes) -> Path:
        relative_path = sanitize_upload_relative_path(filename)
        target_dir = event_dir / relative_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        candidate = target_dir / relative_path.name
        if candidate.exists():
            candidate = target_dir / f"{candidate.stem}_{uuid4().hex[:8]}{candidate.suffix.lower()}"

        candidate.write_bytes(content)
        return candidate

    def _reset_face_crops(self) -> None:
        if self.settings.face_crop_dir.exists():
            shutil.rmtree(self.settings.face_crop_dir)
        self.settings.face_crop_dir.mkdir(parents=True, exist_ok=True)

    def _is_runtime_import_path(self, image_path: Path) -> bool:
        parts = image_path.relative_to(self.settings.images_dir).parts
        return len(parts) >= 2 and parts[0] == "_events"


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
        return is_supported_image_name(filename)

    def search(self, query_path: Path, event_id: int | None = None) -> dict[str, object]:
        query_width, query_height, query_faces = detect_faces_for_path(
            image_path=query_path,
            max_image_size=self.settings.max_image_size,
            primary_engine=self.primary_engine,
            verify_engine=self.verify_engine,
            small_face_threshold=self.settings.small_face_threshold,
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
            matches = self._find_matches(
                query_embeddings=face.embeddings,
                query_context=face.context_features,
                event_id=event_id,
            )
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
            "event_id": event_id,
        }

    def search_many(
        self,
        uploaded_files: list[tuple[str, bytes]],
        event_id: int | None = None,
    ) -> dict[str, object]:
        photo_results: list[dict[str, object]] = []

        for filename, content in uploaded_files:
            if not content or not self.is_supported_image(filename):
                continue

            stored_path = self.store_query_file(filename, content)
            result = self.search(stored_path, event_id=event_id)
            photo_results.append(
                {
                    "filename": filename,
                    "query_path": result["query_path"],
                    "query_width": result["query_width"],
                    "query_height": result["query_height"],
                    "query_faces": result["query_faces"],
                }
            )

        return {
            "photos": photo_results,
            "photo_count": len(photo_results),
        }

    def search_indexed_face(self, face_id: int, event_id: int | None = None) -> dict[str, object] | None:
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
                query_embeddings=query_embeddings,
                query_context=json.loads(query_face["context_json"] or "{}"),
                exclude_face_ids={int(query_face["id"])},
                event_id=event_id,
            ),
        }

        return {
            "query_path": query_face["relative_path"],
            "query_width": int(query_face["width"]),
            "query_height": int(query_face["height"]),
            "query_faces": [query_face_result],
            "source_photo_id": int(query_face["photo_id"]),
            "source_relative_path": query_face["relative_path"],
            "source_event_id": int(query_face["event_id"]),
        }

    def search_person(self, person_id: int, event_id: int | None = None) -> dict[str, object] | None:
        person = self.repository.get_person(person_id)
        if person is None:
            return None

        confirmed_faces_all = self.repository.get_person_faces(person_id)
        if not confirmed_faces_all:
            return None

        confirmed_faces_for_scope = self.repository.get_person_faces(person_id, event_id=event_id)
        aggregated_embeddings = aggregate_person_embeddings(
            self.repository.get_person_profile_embeddings(person_id)
        )
        aggregated_context = aggregate_person_contexts(
            self.repository.get_person_contexts(person_id)
        )
        matches = self._find_matches(
            query_embeddings=aggregated_embeddings,
            query_context=aggregated_context,
            exclude_face_ids={int(face["id"]) for face in confirmed_faces_all},
            event_id=event_id,
        )
        gallery = merge_person_gallery(
            confirmed_faces=confirmed_faces_for_scope,
            matches=matches,
        )

        return {
            "person": {
                "id": int(person["id"]),
                "display_name": str(person["display_name"]),
                "confirmed_face_count": int(person["confirmed_face_count"]),
                "photo_count": int(person["photo_count"]),
            },
            "confirmed_faces": [build_confirmed_face_item(face) for face in confirmed_faces_for_scope],
            "photos": gallery,
            "event_id": event_id,
        }

    def _find_matches(
        self,
        query_embeddings: dict[str, list[float]],
        query_context: dict[str, object] | None,
        exclude_face_ids: set[int] | None = None,
        event_id: int | None = None,
    ) -> list[dict[str, object]]:
        primary_embedding = query_embeddings.get(self.primary_engine.name)
        verify_embedding = query_embeddings.get(self.verify_engine.name)
        if primary_embedding is None or verify_embedding is None:
            return []

        indexed_faces = self.repository.list_indexed_faces_for_engine(
            self.primary_engine.name,
            event_id=event_id,
        )
        if not indexed_faces:
            return []

        primary_vectors = np.array(
            [json.loads(row["embedding_json"]) for row in indexed_faces],
            dtype=np.float32,
        )
        primary_scores = cosine_similarity(
            primary_vectors,
            np.array(primary_embedding, dtype=np.float32),
        )
        ranked = sorted(enumerate(primary_scores), key=lambda item: float(item[1]), reverse=True)

        excluded_ids = exclude_face_ids or set()
        candidate_rows: list[tuple[object, float]] = []
        for row_index, primary_score in ranked:
            row = indexed_faces[row_index]
            if int(row["id"]) in excluded_ids:
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
            candidate_context = json.loads(row["context_json"]) if row["context_json"] else {}
            context_score, context_details = compute_context_score(query_context, candidate_context)
            final_score = combined_match_score(
                primary_score=primary_score,
                verify_score=verify_score,
                verify_threshold=self.settings.dlib_threshold,
                context_score=context_score,
            )
            confidence = classify_match(
                primary_score=primary_score,
                verify_score=verify_score,
                context_score=context_score,
                primary_threshold=self.settings.insightface_threshold,
                verify_threshold=self.settings.dlib_threshold,
                is_small_face=is_small_face_pair(
                    query_context=query_context,
                    candidate_context=candidate_context,
                    small_face_threshold=self.settings.small_face_threshold,
                ),
                final_score=final_score,
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
                "context_score": float(context_score),
                "final_score": float(final_score),
                "context_details": context_details,
                "person_id": int(row["person_id"]) if row["person_id"] is not None else None,
                "person_name": str(row["person_name"]) if row["person_name"] else None,
                "status_label": build_confidence_label(confidence),
                "status_kind": "match",
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
    small_face_threshold: int,
) -> tuple[int, int, list[DetectedFace]]:
    image_array, scale, original_width, original_height = load_image_array(image_path, max_image_size)
    original_image_array = load_original_image_array(image_path)
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

    if faces:
        enrich_face_embeddings(
            image_array=original_image_array,
            faces=faces,
            engines=[primary_engine, verify_engine],
            small_face_threshold=small_face_threshold,
        )
        contexts = build_face_context_features(original_image_array, faces)
        for face, context in zip(faces, contexts):
            face.context_features = context

    return original_width, original_height, faces


def sanitize_upload_relative_path(filename: str) -> Path:
    relative = PurePosixPath(filename or f"upload_{uuid4().hex}.jpg")
    safe_parts = [part for part in relative.parts if part not in {"", ".", ".."}]
    if not safe_parts:
        safe_parts = [f"upload_{uuid4().hex}.jpg"]

    candidate = Path(*safe_parts)
    suffix = candidate.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        candidate = candidate.with_suffix(".jpg")
    return candidate


def is_supported_image_name(filename: str | None) -> bool:
    suffix = Path(filename or "").suffix.lower()
    return suffix in {".jpg", ".jpeg", ".png", ".webp"}


def aggregate_person_embeddings(
    embeddings_by_engine: dict[str, list[list[float]]],
) -> dict[str, list[float]]:
    aggregated: dict[str, list[float]] = {}
    for engine, embeddings in embeddings_by_engine.items():
        vectors = []
        for embedding in embeddings:
            vector = normalize_query_vector(np.array(embedding, dtype=np.float32))
            if vector is not None:
                vectors.append(vector)
        if not vectors:
            continue
        centroid = np.mean(np.stack(vectors, axis=0), axis=0)
        normalized = normalize_query_vector(centroid)
        if normalized is not None:
            aggregated[engine] = [float(value) for value in normalized.tolist()]
    return aggregated


def aggregate_person_contexts(contexts: list[dict[str, object]]) -> dict[str, object] | None:
    if not contexts:
        return None

    vector_keys = [
        "position_vector",
        "neighbor_vector",
        "clothing_histogram",
        "texture_vector",
    ]
    aggregated: dict[str, object] = {}
    for key in vector_keys:
        values = [
            np.array(context.get(key, []), dtype=np.float32)
            for context in contexts
            if context.get(key)
        ]
        if not values:
            continue
        width = max(int(value.shape[0]) for value in values)
        padded = []
        for value in values:
            if int(value.shape[0]) < width:
                value = np.pad(value, (0, width - int(value.shape[0])))
            padded.append(value)
        aggregated[key] = [float(item) for item in np.mean(np.stack(padded, axis=0), axis=0).tolist()]

    if not aggregated:
        return None

    face_widths = [int(context.get("face_width_px", 0)) for context in contexts if context.get("face_width_px")]
    face_heights = [int(context.get("face_height_px", 0)) for context in contexts if context.get("face_height_px")]
    aggregated["face_width_px"] = int(round(sum(face_widths) / max(1, len(face_widths)))) if face_widths else 0
    aggregated["face_height_px"] = int(round(sum(face_heights) / max(1, len(face_heights)))) if face_heights else 0
    return aggregated


def merge_person_gallery(
    confirmed_faces: list[sqlite3_like_row],
    matches: list[dict[str, object]],
) -> list[dict[str, object]]:
    gallery: dict[int, dict[str, object]] = {}
    for face in confirmed_faces:
        photo_id = int(face["photo_id"])
        gallery[photo_id] = {
            "photo_id": photo_id,
            "relative_path": face["relative_path"],
            "width": int(face["width"]),
            "height": int(face["height"]),
            "face_id": int(face["id"]),
            "person_index": int(face["person_index"]),
            "label": build_indexed_face_label(int(face["id"])),
            "top_px": int(face["top_px"]),
            "right_px": int(face["right_px"]),
            "bottom_px": int(face["bottom_px"]),
            "left_px": int(face["left_px"]),
            "crop_path": face["crop_path"],
            "confidence": "high",
            "confidence_label": "Подтверждено",
            "status_label": "Подтверждено",
            "status_kind": "confirmed",
        }

    for match in matches:
        photo_id = int(match["photo_id"])
        gallery.setdefault(photo_id, match)

    return sorted(
        gallery.values(),
        key=lambda item: (
            0 if item["status_kind"] == "confirmed" else 1,
            str(item["relative_path"]),
        ),
    )


def build_confirmed_face_item(face: sqlite3_like_row) -> dict[str, object]:
    return {
        "face_id": int(face["id"]),
        "photo_id": int(face["photo_id"]),
        "label": build_indexed_face_label(int(face["id"])),
        "person_index": int(face["person_index"]),
        "top_px": int(face["top_px"]),
        "right_px": int(face["right_px"]),
        "bottom_px": int(face["bottom_px"]),
        "left_px": int(face["left_px"]),
        "crop_path": face["crop_path"],
        "relative_path": face["relative_path"],
        "width": int(face["width"]),
        "height": int(face["height"]),
        "event_id": int(face["event_id"]),
        "event_name": str(face["event_name"]),
        "event_date": str(face["event_date"] or ""),
        "status_label": "Подтверждено",
    }


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
    context_score: float,
    primary_threshold: float,
    verify_threshold: float,
    is_small_face: bool,
    final_score: float,
) -> str:
    strong_primary_threshold = min(0.99, primary_threshold + 0.1)
    relaxed_verify_threshold = verify_threshold + 0.05

    if primary_score >= strong_primary_threshold and verify_score <= verify_threshold:
        return "high"
    if primary_score >= primary_threshold and verify_score <= relaxed_verify_threshold:
        return "medium"
    if (
        is_small_face
        and primary_score >= (primary_threshold - 0.08)
        and verify_score <= (verify_threshold + 0.22)
        and context_score >= 0.62
        and final_score >= 0.64
    ):
        return "medium"
    if (
        primary_score >= (primary_threshold - 0.03)
        and verify_score <= (verify_threshold + 0.1)
        and context_score >= 0.78
        and final_score >= 0.7
    ):
        return "medium"
    return "rejected"


def match_sort_key(match: dict[str, object]) -> tuple[float, float, float, float]:
    confidence_rank = 0 if match["confidence"] == "high" else 1
    return (
        float(confidence_rank),
        -float(match["final_score"]),
        -float(match["primary_score"]),
        float(match["verify_score"]),
    )


def build_crop_filename(base_name: str, person_index: int) -> str:
    sanitized = base_name.replace("/", "_").replace(" ", "_")
    return f"{sanitized}_person_{person_index}.jpg"


def build_indexed_face_label(face_id: int) -> str:
    return f"Лицо {face_id}"


def build_confidence_label(confidence: str) -> str:
    return "Высокое совпадение" if confidence == "high" else "Среднее совпадение"


def combined_match_score(
    primary_score: float,
    verify_score: float,
    verify_threshold: float,
    context_score: float,
) -> float:
    primary_component = max(0.0, min(1.0, (primary_score + 1.0) / 2.0))
    verify_component = max(0.0, 1.0 - (verify_score / max(verify_threshold + 0.3, 0.01)))
    return float((0.68 * primary_component) + (0.20 * verify_component) + (0.12 * context_score))


def is_small_face_pair(
    query_context: dict[str, object] | None,
    candidate_context: dict[str, object] | None,
    small_face_threshold: int,
) -> bool:
    if not query_context or not candidate_context:
        return False

    query_min_side = min(
        int(query_context.get("face_width_px", small_face_threshold)),
        int(query_context.get("face_height_px", small_face_threshold)),
    )
    candidate_min_side = min(
        int(candidate_context.get("face_width_px", small_face_threshold)),
        int(candidate_context.get("face_height_px", small_face_threshold)),
    )
    return min(query_min_side, candidate_min_side) < small_face_threshold


def normalize_query_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


sqlite3_like_row = object
