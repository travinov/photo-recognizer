from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from photo_recognizer.models import DetectedFace


class FaceEngine(ABC):
    name: str
    embedding_version: str = "1"

    @abstractmethod
    def detect_faces(
        self,
        image_array: np.ndarray,
        scale: float,
        original_width: int,
        original_height: int,
    ) -> list[DetectedFace]:
        raise NotImplementedError

    @abstractmethod
    def embed_faces(
        self,
        image_array: np.ndarray,
        scale: float,
        original_width: int,
        original_height: int,
        faces: list[DetectedFace],
    ) -> list[list[float] | None]:
        raise NotImplementedError

    @abstractmethod
    def embed_face_crop(self, image_array: np.ndarray) -> list[float] | None:
        raise NotImplementedError
