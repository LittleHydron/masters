"""Класичний контурний детектор (baseline без навчання).

Це опорна точка порівняння із задачі 1: він не потребує датасету
й навчання, працює на CPU, але чутливий до фону та освітлення.
Саме цю чутливість і треба показати числами проти YOLO.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import BaseDetector, Detection, TargetShape


class OpenCVDetector(BaseDetector):
    """Детекція обруча/рамки через контурний аналіз.

    Пайплайн: сірий -> згладжування -> Canny -> морфологічне замикання
    -> контури -> геометрична фільтрація. Морфологія тут не косметика:
    обруч часто дає розірваний контур на тлі зі складною текстурою,
    і без замикання findContours ріже його на дуги.
    """

    name = "opencv_contour"
    supported_shapes = (TargetShape.CIRCLE, TargetShape.RECTANGLE)

    def __init__(
        self,
        shape: TargetShape = TargetShape.CIRCLE,
        min_confidence: float = 0.25,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area_ratio: float = 0.002,
        max_area_ratio: float = 0.9,
        blur_kernel: int = 5,
        close_kernel: int = 5,
    ) -> None:
        super().__init__(min_confidence=min_confidence)
        self.shape = shape
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.blur_kernel = blur_kernel
        self.close_kernel = close_kernel

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel)
        )
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    def _detect_impl(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        edges = self._preprocess(frame)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        results: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            ratio = area / frame_area
            if not (self.min_area_ratio <= ratio <= self.max_area_ratio):
                continue

            if self.shape is TargetShape.CIRCLE:
                det = self._fit_circle(contour, area)
            else:
                det = self._fit_rectangle(contour, area)

            if det is not None:
                results.append(det)

        return results

    def _fit_circle(self, contour: np.ndarray, area: float) -> Detection | None:
        """Підгонка еліпса. Впевненість = наскільки контур схожий на еліпс."""
        if len(contour) < 5:
            return None

        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (axis_a, axis_b), angle = ellipse

        if axis_a <= 0 or axis_b <= 0:
            return None

        ellipse_area = np.pi * (axis_a / 2.0) * (axis_b / 2.0)
        if ellipse_area <= 0:
            return None

        perimeter = cv2.arcLength(contour, True)
        
        a, b = axis_a / 2.0, axis_b / 2.0
        h_ram = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0.0
        ellipse_perim = np.pi * (a + b) * (1 + (3 * h_ram) / (10 + np.sqrt(4 - 3 * h_ram)))
        if ellipse_perim <= 0:
            return None

        score_single = 1.0 - min(1.0, abs(perimeter - ellipse_perim) / ellipse_perim)
        score_double = 1.0 - min(1.0, abs(perimeter - 2.0 * ellipse_perim) / (2.0 * ellipse_perim))
        perim_score = max(score_single, score_double)

        elongation = max(axis_a, axis_b) / max(1e-6, min(axis_a, axis_b))
        elong_score = 1.0 if elongation <= 5.0 else max(0.0, 1.0 - (elongation - 5.0) / 5.0)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = (area / hull_area) if hull_area > 1e-6 else 0.0
        hull_fill = hull_area / ellipse_area if ellipse_area > 0 else 0.0
        fill_score = 1.0 if hull_fill >= 0.6 else max(0.0, hull_fill / 0.6)

        confidence = float(perim_score * elong_score * fill_score)

        rx, ry = axis_a / 2.0, axis_b / 2.0
        rad = np.deg2rad(angle)
        
        ext_x = np.sqrt((rx * np.cos(rad)) ** 2 + (ry * np.sin(rad)) ** 2)
        ext_y = np.sqrt((rx * np.sin(rad)) ** 2 + (ry * np.cos(rad)) ** 2)

        return Detection(
            shape=TargetShape.CIRCLE,
            bbox=(cx - ext_x, cy - ext_y, cx + ext_x, cy + ext_y),
            confidence=confidence,
            ellipse=ellipse,
            meta={
                "elongation": float(elongation),
                "perim_score": float(perim_score),
                "hull_fill": float(hull_fill),
                "contour_area": float(area),
            },
        )

    def _fit_rectangle(self, contour: np.ndarray, area: float) -> Detection | None:
        """Полігональна апроксимація до 4 кутів."""
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            return None

        corners = approx.reshape(4, 2).astype(np.float64)
        rect_area = cv2.contourArea(approx)
        if rect_area <= 0:
            return None

        (_, _), (rw, rh), _ = cv2.minAreaRect(approx)
        box_area = rw * rh
        fill_score = min(1.0, rect_area / box_area) if box_area > 0 else 0.0

        aspect = max(rw, rh) / max(1e-6, min(rw, rh))
        aspect_score = 1.0 if aspect <= 3.0 else max(0.0, 1.0 - (aspect - 3.0) / 3.0)

        confidence = float(fill_score * aspect_score)

        xs, ys = corners[:, 0], corners[:, 1]
        return Detection(
            shape=TargetShape.RECTANGLE,
            bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
            confidence=confidence,
            corners=order_corners(corners),
            meta={"aspect": float(aspect), "fill_score": float(fill_score)},
        )


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Впорядкувати 4 кути як TL, TR, BR, BL.

    Без цього solvePnP зіставить кути зображення з невідповідними
    точками моделі й видасть валідну на вигляд, але хибну позу.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float64)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered
