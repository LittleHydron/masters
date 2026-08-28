"""Згенерувати синтетичне відео підльоту до обруча.

Потрібне, щоб перевірити контролери до появи дрона й реального
відео: траєкторія відома, тож видно, чи команди осмислені.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def render(path: str, w=960, h=720, fps=30, n=240, diameter=0.60,
           start=(1.2, 0.45, 5.0), tilt_deg=28.0) -> None:
    """Дрон летить по прямій з бічним зсувом; обруч нахилений.

    Так контролери мають спершу вирівнятись, а потім летіти вперед.
    """
    fov_x = np.deg2rad(82.0)
    fx = (w / 2.0) / np.tan(fov_x / 2.0)
    K = np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])

    radius = diameter / 2.0
    tilt = np.deg2rad(tilt_deg)
    normal = np.array([np.sin(tilt), 0.0, -np.cos(tilt)])
    normal /= np.linalg.norm(normal)

    helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    theta = np.linspace(0, 2 * np.pi, 220, endpoint=False)
    ring = radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    sx, sy, sz = start
    for i in range(n):
        s = i / max(1, n - 1)
        # Обруч стоїть у нулі; камера підлітає до нього вздовж -z,
        # поступово гасячи бічний зсув. Позиція обруча в системі
        # камери = ring - cam, тому cam.z має бути від'ємним.
        cam = np.array([sx * (1 - s) ** 1.5, sy * (1 - s) ** 1.5,
                        -(sz * (1 - s) + 0.35 * s)])

        img = np.full((h, w, 3), 38, np.uint8)
        # Фон із текстурою — щоб контурний детектор мав завади,
        # інакше тест був би нечесно легким.
        rng = np.random.default_rng(i)
        for _ in range(45):
            x, y = rng.integers(0, w), rng.integers(0, h)
            cv2.circle(img, (int(x), int(y)), int(rng.integers(4, 22)),
                       tuple(int(c) for c in rng.integers(45, 75, 3)), -1)
        cv2.line(img, (0, int(h * 0.82)), (w, int(h * 0.82)), (60, 60, 60), 3)

        pts3d = ring - cam
        if np.all(pts3d[:, 2] > 0.25):
            p = (K @ pts3d.T).T
            p = (p[:, :2] / p[:, 2:3]).astype(np.int32)
            cv2.polylines(img, [p.reshape(-1, 1, 2)], True, (40, 180, 245), 13)
            cv2.polylines(img, [p.reshape(-1, 1, 2)], True, (25, 120, 200), 3)

        writer.write(img)

    writer.release()
    print(f"Записано {n} кадрів -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/synthetic_flight.mp4")
    ap.add_argument("--frames", type=int, default=240)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    render(a.out, n=a.frames)
