from __future__ import annotations

from photo_recognizer.config import Settings

from .base import FaceEngine
from .dlib_engine import DlibFaceEngine
from .insightface_engine import InsightFaceEngine


def build_engine(name: str, settings: Settings) -> FaceEngine:
    normalized = name.strip().lower()
    if normalized == "dlib":
        return DlibFaceEngine(
            detection_model=settings.detection_model,
            detection_upsample=settings.detection_upsample,
        )
    if normalized == "insightface":
        return InsightFaceEngine(
            model_name=settings.insightface_model,
            det_size=settings.insightface_det_size,
            root=settings.insightface_root,
        )
    raise ValueError(f"Unsupported face engine: {name}")


__all__ = ["FaceEngine", "build_engine"]
