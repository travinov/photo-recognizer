from __future__ import annotations

from pathlib import Path

import numpy as np

from photo_recognizer.face_engine import scale_coordinate, scaled_location_for_face
from photo_recognizer.models import DetectedFace

from .base import FaceEngine

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover
    FaceAnalysis = None


class InsightFaceEngine(FaceEngine):
    name = "insightface"

    def __init__(self, model_name: str, det_size: int, root: Path) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self.root = root
        self.embedding_version = f"{model_name}-1"
        self._analysis: FaceAnalysis | None = None

    def detect_faces(
        self,
        image_array: np.ndarray,
        scale: float,
        original_width: int,
        original_height: int,
    ) -> list[DetectedFace]:
        detections = self._get_analysis().get(image_array)
        ordered_detections = sorted(detections, key=lambda face: (float(face.bbox[0]), float(face.bbox[1])))

        faces: list[DetectedFace] = []
        for person_index, detection in enumerate(ordered_detections, start=1):
            left, top, right, bottom = self._normalized_bbox(
                detection.bbox,
                resized_width=int(image_array.shape[1]),
                resized_height=int(image_array.shape[0]),
            )
            embedding = self._extract_embedding(detection)
            face = DetectedFace(
                person_index=person_index,
                top=scale_coordinate(top, scale, original_height),
                right=scale_coordinate(right, scale, original_width),
                bottom=scale_coordinate(bottom, scale, original_height),
                left=scale_coordinate(left, scale, original_width),
                embeddings={self.name: embedding} if embedding is not None else {},
            )
            faces.append(face)

        return faces

    def embed_faces(
        self,
        image_array: np.ndarray,
        scale: float,
        original_width: int,
        original_height: int,
        faces: list[DetectedFace],
    ) -> list[list[float] | None]:
        del original_width, original_height
        detections = self._get_analysis().get(image_array)
        if not detections:
            return [None] * len(faces)

        resized_height = int(image_array.shape[0])
        resized_width = int(image_array.shape[1])
        detected_boxes = [
            self._normalized_bbox(detection.bbox, resized_width=resized_width, resized_height=resized_height)
            for detection in detections
        ]

        embeddings: list[list[float] | None] = []
        for face in faces:
            target_top, target_right, target_bottom, target_left = scaled_location_for_face(
                face,
                scale,
                resized_height,
                resized_width,
            )
            target_box = (target_left, target_top, target_right, target_bottom)
            best_index = -1
            best_iou = 0.0

            for index, candidate_box in enumerate(detected_boxes):
                score = box_iou(target_box, candidate_box)
                if score > best_iou:
                    best_iou = score
                    best_index = index

            if best_index == -1 or best_iou < 0.2:
                embeddings.append(None)
                continue

            embeddings.append(self._extract_embedding(detections[best_index]))

        return embeddings

    def _get_analysis(self) -> FaceAnalysis:
        require_insightface()
        if self._analysis is None:
            try:
                self._analysis = FaceAnalysis(
                    name=self.model_name,
                    root=str(self.root),
                    providers=["CPUExecutionProvider"],
                )
                self._analysis.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "InsightFace model files are unavailable. "
                    f"Expected model '{self.model_name}' under '{self.root / 'models'}' "
                    "or downloadable from GitHub on first run. "
                    "If the host is offline, pre-download the model and set "
                    "PHOTO_RECOGNIZER_INSIGHTFACE_ROOT to that directory."
                ) from exc
        return self._analysis

    def _extract_embedding(self, detection: object) -> list[float] | None:
        embedding = getattr(detection, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(detection, "embedding", None)
        if embedding is None:
            return None

        vector = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return [float(value) for value in vector.tolist()]

    def _normalized_bbox(
        self,
        bbox: object,
        resized_width: int,
        resized_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        left = max(0, min(resized_width, x1))
        top = max(0, min(resized_height, y1))
        right = max(left, min(resized_width, x2))
        bottom = max(top, min(resized_height, y2))
        return left, top, right, bottom


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])

    if right <= left or bottom <= top:
        return 0.0

    intersection = float((right - left) * (bottom - top))
    first_area = float(max(0, first[2] - first[0]) * max(0, first[3] - first[1]))
    second_area = float(max(0, second[2] - second[0]) * max(0, second[3] - second[1]))
    denominator = first_area + second_area - intersection
    if denominator <= 0:
        return 0.0
    return intersection / denominator


def require_insightface() -> None:
    if FaceAnalysis is None:
        raise RuntimeError(
            "The 'insightface' package is not installed. "
            "Install dependencies with 'pip install -r requirements.txt'."
        )
