from __future__ import annotations

import numpy as np

from photo_recognizer.face_engine import scale_coordinate, scaled_location_for_face
from photo_recognizer.models import DetectedFace

from .base import FaceEngine

try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None


class DlibFaceEngine(FaceEngine):
    name = "dlib"
    embedding_version = "face_recognition-1"

    def __init__(self, detection_model: str, detection_upsample: int) -> None:
        self.detection_model = detection_model
        self.detection_upsample = detection_upsample

    def detect_faces(
        self,
        image_array: np.ndarray,
        scale: float,
        original_width: int,
        original_height: int,
    ) -> list[DetectedFace]:
        require_face_recognition()

        locations = face_recognition.face_locations(
            image_array,
            number_of_times_to_upsample=self.detection_upsample,
            model=self.detection_model,
        )
        ordered_locations = sorted(locations, key=lambda location: (location[3], location[0]))

        faces: list[DetectedFace] = []
        for person_index, location in enumerate(ordered_locations, start=1):
            top, right, bottom, left = location
            face = DetectedFace(
                person_index=person_index,
                top=scale_coordinate(top, scale, original_height),
                right=scale_coordinate(right, scale, original_width),
                bottom=scale_coordinate(bottom, scale, original_height),
                left=scale_coordinate(left, scale, original_width),
            )
            embedding = self._encode_location(image_array, location)
            if embedding is not None:
                face.embeddings[self.name] = embedding
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
        resized_height = int(image_array.shape[0])
        resized_width = int(image_array.shape[1])
        return [
            self._encode_location(
                image_array,
                scaled_location_for_face(face, scale, resized_height, resized_width),
            )
            for face in faces
        ]

    def _encode_location(
        self,
        image_array: np.ndarray,
        location: tuple[int, int, int, int],
    ) -> list[float] | None:
        encodings = face_recognition.face_encodings(image_array, known_face_locations=[location])
        if not encodings:
            return None
        return [float(value) for value in encodings[0].tolist()]


def require_face_recognition() -> None:
    if face_recognition is None:
        raise RuntimeError(
            "The 'face_recognition' package is not installed. "
            "Install dependencies with 'pip install -r requirements.txt'."
        )
