from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_real_backgrounds(folder: str | Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)


def procedural_room(w: int, h: int, rng: np.random.Generator,
                    distractor_circles: bool = True) -> np.ndarray:
    r = rng.random()
    if r < 0.40:
        hue = rng.uniform(5, 35)
        sat = rng.uniform(25, 110)
        val = rng.uniform(140, 240)
    elif r < 0.55:
        hue = rng.uniform(80, 130)
        sat = rng.uniform(20, 90)
        val = rng.uniform(110, 225)
    else:
        hue = rng.uniform(0, 180)
        sat = rng.uniform(0, 30)
        val = rng.uniform(95, 220)
    wall = cv2.cvtColor(
        np.uint8([[[int(hue), int(sat), int(val)]]]),
        cv2.COLOR_HSV2BGR)[0, 0].astype(np.float64) + rng.normal(0, 8, 3)
    wall = np.clip(wall, 40, 245)
    img = np.ones((h, w, 3), np.float64) * wall

    grad = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.15), h)
    img *= grad[:, None, None]

    horizon = int(h * rng.uniform(0.55, 0.85))
    fg = rng.uniform(60, 170)
    floor = np.clip(fg + rng.normal(0, 18, 3), 30, 200)
    img[horizon:] = floor * rng.uniform(0.85, 1.1)

    cv2.line(img, (0, horizon), (w, horizon),
             tuple(float(c) for c in np.clip(floor * 0.6, 0, 255)),
             max(1, int(h * 0.006)))

    for _ in range(rng.integers(2, 8)):
        x1 = int(rng.uniform(0, w)); y1 = int(rng.uniform(0, h))
        x2 = int(np.clip(x1 + rng.uniform(0.05, 0.45) * w, 0, w))
        y2 = int(np.clip(y1 + rng.uniform(0.05, 0.5) * h, 0, h))
        g = rng.uniform(45, 200)
        col = np.clip(g + rng.normal(0, 22, 3), 25, 240)
        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(float(c) for c in col), -1)

    for _ in range(rng.integers(3, 14)):
        p1 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        p2 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        g = rng.uniform(40, 210)
        col = np.clip(g + rng.normal(0, 18, 3), 20, 240)
        cv2.line(img, p1, p2, tuple(float(c) for c in col),
                 int(rng.integers(1, 5)))

    for _ in range(rng.integers(0, 4) if distractor_circles else 0):
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        ax = (int(rng.uniform(10, w * 0.2)), int(rng.uniform(10, h * 0.2)))
        g = rng.uniform(50, 215)
        col = np.clip(g + rng.normal(0, 20, 3), 25, 240)
        cv2.ellipse(img, c, ax, float(rng.uniform(0, 180)), 0, 360,
                    tuple(float(x) for x in col), int(rng.integers(1, 6)))

    if rng.random() < 0.35:
        x1 = int(rng.uniform(0, w * 0.7)); y1 = int(rng.uniform(0, h * 0.3))
        x2 = int(min(w, x1 + rng.uniform(0.15, 0.35) * w))
        y2 = int(min(h, y1 + rng.uniform(0.2, 0.5) * h))
        cv2.rectangle(img, (x1, y1), (x2, y2), (245, 245, 250), -1)

    return np.clip(img, 0, 255).astype(np.uint8)


def prepare_background(path: Path | None, w: int, h: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Взяти реальне фото (з випадковим кропом) або згенерувати."""
    if path is not None:
        img = cv2.imread(str(path))
        if img is not None:
            ih, iw = img.shape[:2]
            scale = max(w / iw, h / ih) * rng.uniform(1.0, 1.35)
            img = cv2.resize(img, (max(w, int(iw * scale)), max(h, int(ih * scale))))
            ih, iw = img.shape[:2]
            x = int(rng.uniform(0, max(1, iw - w)))
            y = int(rng.uniform(0, max(1, ih - h)))
            return img[y:y + h, x:x + w].copy()
    return procedural_room(w, h, rng)
