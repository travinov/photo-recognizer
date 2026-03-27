from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from photo_recognizer.models import DetectedFace


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def iter_image_files(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


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


def resize_coordinate(value: int, scale: float, max_value: int) -> int:
    if scale == 1.0:
        return value
    scaled = int(round(value * scale))
    return max(0, min(max_value, scaled))


def scaled_location_for_face(
    face: DetectedFace,
    scale: float,
    resized_height: int,
    resized_width: int,
) -> tuple[int, int, int, int]:
    return (
        resize_coordinate(face.top, scale, resized_height),
        resize_coordinate(face.right, scale, resized_width),
        resize_coordinate(face.bottom, scale, resized_height),
        resize_coordinate(face.left, scale, resized_width),
    )
