from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from photo_recognizer.models import DetectedFace

try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def iter_image_files(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def detect_faces(image_path: Path, detection_model: str, max_image_size: int) -> tuple[int, int, list[DetectedFace]]:
    require_face_recognition()

    image_array, scale, original_width, original_height = load_image_array(image_path, max_image_size)
    locations = face_recognition.face_locations(image_array, model=detection_model)
    ordered_locations = sorted(locations, key=lambda location: (location[3], location[0]))
    encodings = face_recognition.face_encodings(image_array, known_face_locations=ordered_locations)

    faces: list[DetectedFace] = []
    for person_index, (location, encoding) in enumerate(zip(ordered_locations, encodings), start=1):
        top, right, bottom, left = location
        faces.append(
            DetectedFace(
                person_index=person_index,
                top=scale_coordinate(top, scale, original_height),
                right=scale_coordinate(right, scale, original_width),
                bottom=scale_coordinate(bottom, scale, original_height),
                left=scale_coordinate(left, scale, original_width),
                embedding=[float(value) for value in encoding.tolist()],
            )
        )

    return original_width, original_height, faces


def save_face_crop(image_path: Path, output_path: Path, top: int, right: int, bottom: int, left: int) -> None:
    with Image.open(image_path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        width, height = normalized.size
        margin = 30
        crop_box = (
            max(0, left - margin),
            max(0, top - margin),
            min(width, right + margin),
            min(height, bottom + margin),
        )
        normalized.crop(crop_box).save(output_path, format="JPEG", quality=90)


def load_image_array(image_path: Path, max_image_size: int) -> tuple[np.ndarray, float, int, int]:
    with Image.open(image_path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        original_width, original_height = normalized.size
        longest_side = max(original_width, original_height)
        scale = 1.0

        if longest_side > max_image_size:
            scale = max_image_size / float(longest_side)
            resized = normalized.resize(
                (
                    max(1, int(original_width * scale)),
                    max(1, int(original_height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        else:
            resized = normalized

    return np.array(resized), scale, original_width, original_height


def scale_coordinate(value: int, scale: float, max_value: int) -> int:
    if scale == 1.0:
        return value
    scaled = int(round(value / scale))
    return max(0, min(max_value, scaled))


def require_face_recognition() -> None:
    if face_recognition is None:
        raise RuntimeError(
            "The 'face_recognition' package is not installed. "
            "Install dependencies with 'pip install -r requirements.txt'."
        )
