"""Рендер обруча з відомою 3D-позою на довільний фон.

Мета — навчити модель ігнорувати оточення: той самий обруч
показується в сотнях різних сцен. Розмітка генерується точно
(bbox, еліпс, поза), бо геометрія відома за побудовою.

Модель обруча — тор: коло радіуса R з трубкою радіуса r.
Проєкція трубки враховується, інакше синтетичний обруч виглядав
би нескінченно тонким і модель не навчилась би бачити реальний.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HoopSpec:
    """Фізичні параметри цілі. За замовчуванням — CSP 89 см."""

    diameter: float = 0.89      # м, зовнішній діаметр
    tube_diameter: float = 0.020  # м, товщина трубки
    color_bgr: tuple[int, int, int] = (40, 40, 200)  # насичений червоний

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def tube_radius(self) -> float:
        return self.tube_diameter / 2.0


@dataclass
class RenderResult:
    image: np.ndarray
    bbox: tuple[float, float, float, float]
    ellipse: tuple  # ((cx,cy),(a,b),angle) — зовнішній контур
    position: np.ndarray   # (3,) центр обруча в системі камери
    normal: np.ndarray     # (3,) нормаль площини обруча
    visible_ratio: float   # частка контуру, що потрапила в кадр


def look_at_normal(tilt_deg: float, pan_deg: float) -> np.ndarray:
    """Нормаль площини обруча за кутами нахилу й повороту."""
    t, p = np.deg2rad(tilt_deg), np.deg2rad(pan_deg)
    n = np.array([np.sin(t) * np.cos(p), np.sin(p), -np.cos(t) * np.cos(p)])
    return n / np.linalg.norm(n)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def render_hoop(
    background: np.ndarray,
    K: np.ndarray,
    position: np.ndarray,
    normal: np.ndarray,
    spec: HoopSpec,
    n_samples: int = 512,
    shading: bool = True,
    rng: np.random.Generator | None = None,
) -> RenderResult | None:
    """Намалювати обруч на фоні. Повертає None, якщо ціль поза кадром."""
    rng = rng or np.random.default_rng()
    img = background.copy()
    h, w = img.shape[:2]

    position = np.asarray(position, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    u, v = _plane_basis(normal)

    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    ring = position[None, :] + spec.radius * (
        np.cos(theta)[:, None] * u[None, :] + np.sin(theta)[:, None] * v[None, :]
    )

    if np.any(ring[:, 2] <= 0.15):
        return None  # частина обруча за камерою

    proj = (K @ ring.T).T
    pts = proj[:, :2] / proj[:, 2:3]

    if not np.all(np.isfinite(pts)):
        return None

    # Товщина трубки в пікселях змінюється вздовж контуру: ближчі
    # ділянки товщі. Малюємо посегментно, інакше нахилений обруч
    # виглядав би неправдоподібно.
    fx = float(K[0, 0])
    # Мінімум 2 px: тонша лінія практично не читається після
    # компресії й розмиття, і модель не має що вчити.
    thickness = np.maximum(2.0, spec.tube_radius * 2.0 * fx / ring[:, 2])

    inside = ((pts[:, 0] >= -w * 0.5) & (pts[:, 0] <= w * 1.5)
              & (pts[:, 1] >= -h * 0.5) & (pts[:, 1] <= h * 1.5))
    if inside.sum() < n_samples * 0.25:
        return None

    base = np.array(spec.color_bgr, dtype=np.float64)
    # Освітлення: ділянки, нахилені до камери, світліші.
    light_dir = np.array([0.3, -0.6, -0.75])
    light_dir /= np.linalg.norm(light_dir)

    overlay = img.copy()
    for i in range(n_samples):
        j = (i + 1) % n_samples
        p1 = tuple(int(round(c)) for c in pts[i])
        p2 = tuple(int(round(c)) for c in pts[j])

        if shading:
            # Напрямок від осі обруча до точки — нормаль трубки.
            radial = ring[i] - position
            rn = radial / max(1e-9, np.linalg.norm(radial))
            # Менший діапазон затемнення: сильна тінь робить тонку
            # трубку майже невидимою на темному фоні.
            lam = 0.82 + 0.28 * max(0.0, float(np.dot(rn, -light_dir)))
            color = np.clip(base * lam, 0, 255)
        else:
            color = base

        cv2.line(overlay, p1, p2, tuple(int(c) for c in color),
                 int(round((thickness[i] + thickness[j]) / 2.0)),
                 lineType=cv2.LINE_AA)

    # Легке змішування з фоном: різкий контур виглядає «наклеєним»
    # і модель починає ловити саме різкість краю, а не форму.
    alpha = float(rng.uniform(0.92, 1.0))
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    vis = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    visible_ratio = len(vis) / n_samples
    if len(vis) < 5:
        return None

    pad = float(np.mean(thickness)) / 2.0
    x1 = float(np.clip(vis[:, 0].min() - pad, 0, w - 1))
    y1 = float(np.clip(vis[:, 1].min() - pad, 0, h - 1))
    x2 = float(np.clip(vis[:, 0].max() + pad, 0, w - 1))
    y2 = float(np.clip(vis[:, 1].max() + pad, 0, h - 1))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None

    ellipse = cv2.fitEllipse(pts.astype(np.float32)) if len(pts) >= 5 else None

    return RenderResult(
        image=img, bbox=(x1, y1, x2, y2), ellipse=ellipse,
        position=position, normal=normal, visible_ratio=visible_ratio,
    )
