from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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


def load_original_image_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(normalized)


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


def crop_face_region(
    image_array: np.ndarray,
    face: DetectedFace,
    pad_x_ratio: float,
    pad_top_ratio: float,
    pad_bottom_ratio: float,
) -> np.ndarray:
    height = int(image_array.shape[0])
    width = int(image_array.shape[1])
    face_width = max(1, face.width)
    face_height = max(1, face.height)
    left = max(0, int(round(face.left - (face_width * pad_x_ratio))))
    right = min(width, int(round(face.right + (face_width * pad_x_ratio))))
    top = max(0, int(round(face.top - (face_height * pad_top_ratio))))
    bottom = min(height, int(round(face.bottom + (face_height * pad_bottom_ratio))))
    if right <= left or bottom <= top:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return image_array[top:bottom, left:right].copy()


def upscale_image_array(image_array: np.ndarray, factor: int) -> np.ndarray:
    if image_array.size == 0 or factor <= 1:
        return image_array
    image = Image.fromarray(image_array)
    return np.array(
        image.resize(
            (image.width * factor, image.height * factor),
            Image.Resampling.LANCZOS,
        )
    )


def sharpen_image_array(image_array: np.ndarray) -> np.ndarray:
    if image_array.size == 0:
        return image_array
    image = Image.fromarray(image_array)
    return np.array(image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=180, threshold=2)))


def brighten_image_array(image_array: np.ndarray) -> np.ndarray:
    if image_array.size == 0:
        return image_array
    image = Image.fromarray(image_array)
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.12)
    image = ImageEnhance.Brightness(image).enhance(1.05)
    return np.array(image)
