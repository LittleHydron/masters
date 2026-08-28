"""Деградація відео з телефона до якості камери дрона.

Відео зняте iPhone має різкість і бітрейт, яких у Tello немає:
модель, перевірена на такому матеріалі, дала б завищену оцінку.

Поле зору НЕ чіпаємо (рішення користувача 2026-08-27): у Tello воно
ширше (82.6 град проти виміряних 55.8), тож звузити кроп можна, а
розширити — ні. Точна емуляція неможлива; підганяти будемо за фактом,
коли з'явиться реальний дрон. Тому тест ТРОХИ ЛЕГШИЙ за реальність —
обруч у кадрі більший, ніж побачив би Tello з тієї ж дистанції.

Порядок операцій відповідає фізичному тракту: оптика -> сенсор ->
кодек. Переставляти не можна: компресія має застосовуватись ОСТАННЬОЮ,
інакше подальше розмиття приховає її артефакти.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TelloProfile:
    """Параметри камери DJI Robomaster TT (Tello Talent)."""

    width: int = 1280           # паспортні 720p
    height: int = 720
    blur_sigma: float = 0.8     # дешева оптика
    noise_sigma: float = 4.0    # малий сенсор
    jpeg_quality: int = 65      # H.264 по Wi-Fi ~ таких артефактів
    fps: int = 30


def degrade(frame: np.ndarray, prof: TelloProfile,
            rng: np.random.Generator) -> np.ndarray:
    """Один кадр -> кадр «як з Tello»."""
    h, w = frame.shape[:2]
    if (w, h) != (prof.width, prof.height):
        # AREA — правильний фільтр для зменшення: усереднює, а не
        # семплює, тож не породжує муару на дрібних деталях.
        frame = cv2.resize(frame, (prof.width, prof.height),
                           interpolation=cv2.INTER_AREA)

    if prof.blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), prof.blur_sigma)

    if prof.noise_sigma > 0:
        noise = rng.normal(0, prof.noise_sigma, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if 0 < prof.jpeg_quality < 100:
        ok, enc = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, prof.jpeg_quality])
        if ok:
            frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return frame
