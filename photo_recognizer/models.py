from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DetectedFace:
    person_index: int
    top: int
    right: int
    bottom: int
    left: int
    embedding: list[float]

    @property
    def label(self) -> str:
        return f"Человек {self.person_index}"
