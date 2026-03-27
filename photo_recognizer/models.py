from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DetectedFace:
    person_index: int
    top: int
    right: int
    bottom: int
    left: int
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"Человек {self.person_index}"

    def embedding_for(self, engine: str) -> list[float] | None:
        return self.embeddings.get(engine)
