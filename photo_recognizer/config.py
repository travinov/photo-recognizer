from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    images_dir: Path
    storage_dir: Path
    db_path: Path
    query_upload_dir: Path
    face_crop_dir: Path
    detection_model: str
    max_image_size: int
    match_threshold: float
    top_k_matches: int


def load_settings() -> Settings:
    base_dir = BASE_DIR
    storage_dir = Path(os.getenv("PHOTO_RECOGNIZER_STORAGE_DIR", base_dir / "data")).resolve()
    images_dir = Path(os.getenv("PHOTO_RECOGNIZER_IMAGES_DIR", base_dir / "tcf")).resolve()

    settings = Settings(
        base_dir=base_dir,
        images_dir=images_dir,
        storage_dir=storage_dir,
        db_path=Path(os.getenv("PHOTO_RECOGNIZER_DB_PATH", storage_dir / "face_index.db")).resolve(),
        query_upload_dir=(storage_dir / "query_uploads").resolve(),
        face_crop_dir=(storage_dir / "face_crops").resolve(),
        detection_model=os.getenv("PHOTO_RECOGNIZER_DETECTION_MODEL", "hog"),
        max_image_size=int(os.getenv("PHOTO_RECOGNIZER_MAX_IMAGE_SIZE", "1600")),
        match_threshold=float(os.getenv("PHOTO_RECOGNIZER_MATCH_THRESHOLD", "0.48")),
        top_k_matches=int(os.getenv("PHOTO_RECOGNIZER_TOP_K", "12")),
    )

    ensure_directories(settings)
    return settings


def ensure_directories(settings: Settings) -> None:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.query_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.face_crop_dir.mkdir(parents=True, exist_ok=True)
