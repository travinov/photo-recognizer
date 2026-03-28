from __future__ import annotations

import math
from typing import Any

import numpy as np

from photo_recognizer.engines import FaceEngine
from photo_recognizer.face_engine import (
    brighten_image_array,
    crop_face_region,
    sharpen_image_array,
    upscale_image_array,
)
from photo_recognizer.models import DetectedFace


def enrich_face_embeddings(
    image_array: np.ndarray,
    faces: list[DetectedFace],
    engines: list[FaceEngine],
    small_face_threshold: int,
) -> None:
    for face in faces:
        variant_images = build_face_variant_images(
            image_array=image_array,
            face=face,
            small_face_threshold=small_face_threshold,
        )
        for engine in engines:
            base_embedding = face.embedding_for(engine.name)
            enhanced_embedding = aggregate_variant_embeddings(
                engine=engine,
                base_embedding=base_embedding,
                variant_images=variant_images,
            )
            if enhanced_embedding is not None:
                face.embeddings[engine.name] = enhanced_embedding


def build_face_variant_images(
    image_array: np.ndarray,
    face: DetectedFace,
    small_face_threshold: int,
) -> list[np.ndarray]:
    base_crop = crop_face_region(
        image_array,
        face,
        pad_x_ratio=0.22,
        pad_top_ratio=0.2,
        pad_bottom_ratio=0.28,
    )
    wide_crop = crop_face_region(
        image_array,
        face,
        pad_x_ratio=0.45,
        pad_top_ratio=0.26,
        pad_bottom_ratio=0.65,
    )

    if base_crop.size == 0:
        return []

    is_small = min(face.width, face.height) < small_face_threshold
    if is_small:
        base_crop = upscale_image_array(base_crop, factor=2)
        if wide_crop.size > 0:
            wide_crop = upscale_image_array(wide_crop, factor=2)

    variants = [base_crop]
    if wide_crop.size > 0:
        variants.append(wide_crop)
    variants.append(sharpen_image_array(base_crop))
    variants.append(brighten_image_array(base_crop))
    return variants


def aggregate_variant_embeddings(
    engine: FaceEngine,
    base_embedding: list[float] | None,
    variant_images: list[np.ndarray],
) -> list[float] | None:
    vectors: list[np.ndarray] = []
    if base_embedding is not None:
        vector = normalize_vector(np.array(base_embedding, dtype=np.float32))
        if vector is not None:
            vectors.append(vector)

    for variant in variant_images:
        embedding = engine.embed_face_crop(variant)
        if embedding is None:
            continue
        vector = normalize_vector(np.array(embedding, dtype=np.float32))
        if vector is not None:
            vectors.append(vector)

    if not vectors:
        return None

    merged = np.mean(np.stack(vectors, axis=0), axis=0)
    normalized = normalize_vector(merged)
    if normalized is None:
        return None
    return [float(value) for value in normalized.tolist()]


def build_face_context_features(
    image_array: np.ndarray,
    faces: list[DetectedFace],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    image_height = int(image_array.shape[0])
    image_width = int(image_array.shape[1])

    for face in faces:
        body_crop = crop_face_region(
            image_array,
            face,
            pad_x_ratio=0.75,
            pad_top_ratio=-0.05,
            pad_bottom_ratio=2.2,
        )
        clothing_hist = compute_color_histogram(body_crop)
        texture_vector = compute_texture_signature(body_crop)
        contexts.append(
            {
                "face_width_px": face.width,
                "face_height_px": face.height,
                "position_vector": build_position_vector(face, image_width, image_height),
                "neighbor_vector": build_neighbor_vector(face, faces),
                "clothing_histogram": clothing_hist,
                "texture_vector": texture_vector,
            }
        )

    return contexts


def compute_context_score(
    query_context: dict[str, Any] | None,
    candidate_context: dict[str, Any] | None,
) -> tuple[float, dict[str, float]]:
    if not query_context or not candidate_context:
        return 0.0, {
            "clothing": 0.0,
            "texture": 0.0,
            "position": 0.0,
            "neighbors": 0.0,
        }

    clothing = cosine_similarity(
        np.array(query_context.get("clothing_histogram", []), dtype=np.float32),
        np.array(candidate_context.get("clothing_histogram", []), dtype=np.float32),
    )
    texture = cosine_similarity(
        np.array(query_context.get("texture_vector", []), dtype=np.float32),
        np.array(candidate_context.get("texture_vector", []), dtype=np.float32),
    )
    position = inverse_distance_similarity(
        np.array(query_context.get("position_vector", []), dtype=np.float32),
        np.array(candidate_context.get("position_vector", []), dtype=np.float32),
        scale=1.2,
    )
    neighbors = inverse_distance_similarity(
        np.array(query_context.get("neighbor_vector", []), dtype=np.float32),
        np.array(candidate_context.get("neighbor_vector", []), dtype=np.float32),
        scale=4.0,
    )
    combined = (0.5 * clothing) + (0.25 * texture) + (0.15 * position) + (0.10 * neighbors)
    return float(combined), {
        "clothing": float(clothing),
        "texture": float(texture),
        "position": float(position),
        "neighbors": float(neighbors),
    }


def build_position_vector(face: DetectedFace, image_width: int, image_height: int) -> list[float]:
    center_x = face.left + (face.width / 2.0)
    center_y = face.top + (face.height / 2.0)
    return [
        round(center_x / max(1, image_width), 6),
        round(center_y / max(1, image_height), 6),
        round(face.width / max(1, image_width), 6),
        round(face.height / max(1, image_height), 6),
    ]


def build_neighbor_vector(face: DetectedFace, faces: list[DetectedFace]) -> list[float]:
    target_center_x = face.left + (face.width / 2.0)
    target_center_y = face.top + (face.height / 2.0)
    face_area = max(1.0, float(face.width * face.height))

    neighbors: list[tuple[float, float, float]] = []
    for candidate in faces:
        if candidate.person_index == face.person_index:
            continue
        candidate_center_x = candidate.left + (candidate.width / 2.0)
        candidate_center_y = candidate.top + (candidate.height / 2.0)
        dx = (candidate_center_x - target_center_x) / max(1.0, float(face.width))
        dy = (candidate_center_y - target_center_y) / max(1.0, float(face.height))
        ratio = math.sqrt(max(1.0, float(candidate.width * candidate.height)) / face_area)
        distance = math.sqrt((dx * dx) + (dy * dy))
        neighbors.append((distance, dx, dy, ratio))

    neighbors.sort(key=lambda item: item[0])
    signature: list[float] = []
    for _, dx, dy, ratio in neighbors[:3]:
        signature.extend([round(dx, 6), round(dy, 6), round(ratio, 6)])

    while len(signature) < 9:
        signature.append(0.0)
    return signature


def compute_color_histogram(image_array: np.ndarray) -> list[float]:
    if image_array.size == 0:
        return [0.0] * 96

    image = image_array.astype(np.float32) / 255.0
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    max_channel = np.maximum.reduce([red, green, blue])
    min_channel = np.minimum.reduce([red, green, blue])
    delta = max_channel - min_channel

    hue = np.zeros_like(max_channel)
    mask = delta > 1e-6
    red_mask = mask & (max_channel == red)
    green_mask = mask & (max_channel == green)
    blue_mask = mask & (max_channel == blue)
    hue[red_mask] = ((green[red_mask] - blue[red_mask]) / delta[red_mask]) % 6.0
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / delta[green_mask]) + 2.0
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / delta[blue_mask]) + 4.0
    hue = (hue / 6.0) % 1.0

    saturation = np.divide(
        delta,
        max_channel,
        out=np.zeros_like(delta),
        where=max_channel > 1e-6,
    )
    value = max_channel
    histogram, _ = np.histogramdd(
        np.stack([hue, saturation, value], axis=-1).reshape(-1, 3),
        bins=(8, 4, 3),
        range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    )
    histogram = histogram.astype(np.float32).reshape(-1)
    return normalize_list(histogram)


def compute_texture_signature(image_array: np.ndarray) -> list[float]:
    if image_array.size == 0:
        return [0.0] * 15

    grayscale = (
        (0.299 * image_array[..., 0])
        + (0.587 * image_array[..., 1])
        + (0.114 * image_array[..., 2])
    ).astype(np.float32)
    histogram, _ = np.histogram(grayscale, bins=12, range=(0.0, 255.0))
    histogram = histogram.astype(np.float32)
    gradient_x = np.abs(np.diff(grayscale, axis=1)).mean() if grayscale.shape[1] > 1 else 0.0
    gradient_y = np.abs(np.diff(grayscale, axis=0)).mean() if grayscale.shape[0] > 1 else 0.0
    summary = np.array(
        [
            float(grayscale.mean() / 255.0),
            float(grayscale.std() / 255.0),
            float((gradient_x + gradient_y) / 255.0),
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([histogram, summary], axis=0)
    return normalize_list(vector)


def normalize_list(vector: np.ndarray) -> list[float]:
    normalized = normalize_vector(vector)
    if normalized is None:
        return [0.0] * int(vector.shape[0])
    return [float(value) for value in normalized.tolist()]


def normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 or second.size == 0:
        return 0.0
    first_norm = normalize_vector(first)
    second_norm = normalize_vector(second)
    if first_norm is None or second_norm is None:
        return 0.0
    return float(np.clip(first_norm @ second_norm, 0.0, 1.0))


def inverse_distance_similarity(first: np.ndarray, second: np.ndarray, scale: float) -> float:
    if first.size == 0 or second.size == 0:
        return 0.0
    distance = float(np.linalg.norm(first - second))
    return float(max(0.0, 1.0 - (distance / scale)))
