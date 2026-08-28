"""Візуалізація детекції та команд керування.

Малює два стеки — IBVS зверху, PBVS знизу — щоб на однаковому
кадрі було видно, що саме вимагає кожна стратегія.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..detection.base import Detection
from ..control.base import ControlCommand

PHASE_COLORS = {
    "SEARCH": (80, 80, 220), "HOLD": (80, 160, 220),
    "ALIGN": (60, 200, 240), "ORBIT": (60, 200, 240),
    "APPROACH": (80, 220, 80), "COMMIT": (60, 240, 140),
    "BLIND": (220, 120, 60), "NO_POSE": (120, 120, 120),
    "idle": (160, 160, 160),
}


def draw_detection(img: np.ndarray, det: Detection | None) -> None:
    h, w = img.shape[:2]
    # Перехрестя центру кадру — опорна точка для IBVS.
    cv2.drawMarker(img, (w // 2, h // 2), (200, 200, 200),
                   cv2.MARKER_CROSS, 22, 1)

    if det is None:
        return

    if det.ellipse is not None:
        cv2.ellipse(img, det.ellipse, (0, 235, 255), 2)
    if det.corners is not None:
        pts = det.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (0, 235, 255), 2)

    x1, y1, x2, y2 = [int(v) for v in det.bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 160, 255), 1)

    cx, cy = det.center
    cv2.drawMarker(img, (int(cx), int(cy)), (0, 235, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    # Вектор помилки — те, що гасить контролер.
    cv2.arrowedLine(img, (w // 2, h // 2), (int(cx), int(cy)),
                    (0, 235, 255), 1, tipLength=0.15)


def draw_command_gauges(img: np.ndarray, cmd: ControlCommand, x0: int, y0: int) -> None:
    """Горизонтальні індикатори каналів керування."""
    channels = [("LAT", cmd.lateral), ("FWD", cmd.forward),
                ("VER", cmd.vertical), ("YAW", cmd.yaw)]
    bar_w, bar_h, gap = 110, 11, 17

    for i, (label, value) in enumerate(channels):
        y = y0 + i * gap
        cv2.putText(img, label, (x0, y + 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (210, 210, 210), 1, cv2.LINE_AA)
        bx = x0 + 34
        cv2.rectangle(img, (bx, y), (bx + bar_w, y + bar_h), (70, 70, 70), 1)
        mid = bx + bar_w // 2
        cv2.line(img, (mid, y), (mid, y + bar_h), (110, 110, 110), 1)

        extent = int(np.clip(value, -1, 1) * (bar_w // 2))
        if abs(extent) > 0:
            color = (90, 220, 90) if value > 0 else (90, 140, 240)
            cv2.rectangle(img, (mid, y + 1), (mid + extent, y + bar_h - 1), color, -1)

        cv2.putText(img, f"{value:+.2f}", (bx + bar_w + 7, y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 210, 210), 1, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, det: Detection | None, cmd: ControlCommand,
               title: str, frame_idx: int, t: float) -> np.ndarray:
    img = frame.copy()
    draw_detection(img, det)

    h, w = img.shape[:2]
    # Напівпрозора підкладка, щоб текст читався на будь-якому фоні.
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (270, 132), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    color = PHASE_COLORS.get(cmd.phase, (200, 200, 200))
    cv2.putText(img, title, (9, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, cmd.phase, (9, 41), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2, cv2.LINE_AA)

    draw_command_gauges(img, cmd, 9, 52)

    cv2.putText(img, f"#{frame_idx}  t={t:5.2f}s", (w - 132, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    reason = cmd.reason[:64]
    cv2.putText(img, reason, (9, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (215, 215, 215), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 2)
    return img


def draw_comparison(frame: np.ndarray, det: Detection | None,
                    cmd_ibvs: ControlCommand, cmd_pbvs: ControlCommand,
                    frame_idx: int, t: float) -> np.ndarray:
    top = draw_panel(frame, det, cmd_ibvs, "IBVS", frame_idx, t)
    bottom = draw_panel(frame, det, cmd_pbvs, "PBVS", frame_idx, t)
    return np.vstack([top, bottom])
