from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TargetShape(str, Enum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"


@dataclass
class Detection:
    """Результат детекції цілі на одному кадрі.

    Поля навмисно надлишкові: різні детектори природно віддають різне.
    OpenCV одразу дає еліпс/полігон, YOLO дає bbox (+ маску, якщо seg-модель).
    Розв'язувач пози бере те, що є, і сам вирішує, чи йому вистачає даних.
    """

    shape: TargetShape
    bbox: tuple[float, float, float, float]
    confidence: float

    ellipse: tuple[tuple[float, float], tuple[float, float], float] | None = None
    corners: np.ndarray | None = None
    mask: np.ndarray | None = None

    inference_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class BaseDetector(ABC):
    name: str = "base"
    supported_shapes: tuple[TargetShape, ...] = ()

    def __init__(self, min_confidence: float = 0.25) -> None:
        self.min_confidence = min_confidence
        self._warmed_up = False

    @abstractmethod
    def _detect_impl(self, frame: np.ndarray) -> list[Detection]:
        """Знайти всі кандидати на кадрі BGR. Без вимірювання часу."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Публічний вхід: детекція + єдиний для всіх замір часу."""
        start = time.perf_counter()
        detections = self._detect_impl(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        for det in detections:
            det.inference_ms = elapsed_ms

        return [d for d in detections if d.confidence >= self.min_confidence]

    def detect_best(self, frame: np.ndarray) -> Detection | None:
        """Найкращий кандидат — те, що потрібно пайплайну керування.

        Обираємо не за впевненістю, а за площею: коли в кадрі кілька
        схожих об'єктів, цільовим є найближчий, тобто найбільший.
        """
        detections = self.detect(frame)
        if not detections:
            return None
        return max(detections, key=lambda d: d.area)

    def warmup(self, frame_shape: tuple[int, int] = (720, 960)) -> None:
        """Прогін на порожньому кадрі.

        Обов'язковий перед бенчмарком: перший виклик YOLO тягне за собою
        аллокацію CUDA/лінивий імпорт і на порядок повільніший за решту,
        що інакше отруює середній FPS.
        """
        dummy = np.zeros((*frame_shape, 3), dtype=np.uint8)
        try:
            self._detect_impl(dummy)
        except Exception:
            pass
        self._warmed_up = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
