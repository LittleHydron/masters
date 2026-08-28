"""Шум детектора: перетворити ідеальну геометрію на реалістичну детекцію.

Найбільша нечесність стенду без цього — детектор, що завжди бачить
ціль ідеально. Реальна YOLO поводиться інакше, і саме це виміряно
на holdout-наборі (див. STATUS.md, «Аналіз пропусків»):

* recall падає з дистанцією: 0.667 на 1-2 м -> 0.353 на 4-6 м;
* пропуск — це ПОВНА невидимість, а не зсунутий бокс: частка
  «бачить, але неточно» (IoU 0.1-0.5) = 0.004, медіанний IoU
  знайдених = 0.915.

Тому модель шуму бінарна: або детекція є і вона точна (з невеликим
піксельним тремтінням), або її немає зовсім. Проміжного стану не
вводимо — це суперечило б виміряному.

Ймовірність виявлення інтерполюється за виміряними точками. За межами
діапазону тримаємо крайні значення, щоб не екстраполювати вигадане.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Виміряно на synthetic_holdout моделлю yolov8n_mixed (273 цілі).
_DIST_M = np.array([1.5, 2.5, 3.5, 5.0])
_RECALL = np.array([0.667, 0.617, 0.507, 0.353])


@dataclass
class NoiseParams:
    """Параметри шуму. `enabled=False` повертає ідеальний детектор."""

    enabled: bool = True
    jitter_px: float = 2.5      # тремтіння кутів бокса, пікселі
    recall_scale: float = 1.0   # <1 — гірший детектор, >1 — кращий
    false_positive_rate: float = 0.0  # хибні спрацювання на кадр


def detection_probability(distance: float, scale: float = 1.0) -> float:
    """Ймовірність виявлення на заданій дистанції."""
    p = float(np.interp(distance, _DIST_M, _RECALL))
    return float(np.clip(p * scale, 0.0, 1.0))


def apply(detection, distance: float, rng: np.random.Generator,
          params: NoiseParams):
    """Пропустити ідеальну детекцію крізь модель шуму.

    Повертає детекцію або None (ціль не виявлено).
    """
    if not params.enabled or detection is None:
        return detection

    if rng.random() > detection_probability(distance, params.recall_scale):
        return None

    if params.jitter_px <= 0:
        return detection

    # Тремтіння: незалежний зсув кутів бокса й центру еліпса.
    x1, y1, x2, y2 = detection.bbox
    j = rng.normal(0, params.jitter_px, 4)
    bbox = (x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3])
    if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
        return None

    ell = detection.ellipse
    if ell is not None:
        (cx, cy), (a, b), ang = ell
        e = rng.normal(0, params.jitter_px, 2)
        # Осі тремтять слабше за центр: розмір оцінюється стабільніше.
        s = rng.normal(0, params.jitter_px * 0.5, 2)
        ell = ((cx + e[0], cy + e[1]),
               (max(2.0, a + s[0]), max(2.0, b + s[1])), ang)

    detection.bbox = bbox
    detection.ellipse = ell
    return detection
