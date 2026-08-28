"""Детектор на базі YOLO (v8n / v11n) за спільним інтерфейсом.

Обидві архітектури обгортаються одним класом: різниця лише у файлі
ваг, тому порівняння в задачі 1 лишається чесним — однаковий
препроцесинг, однакові пороги, однаковий замір часу.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import BaseDetector, Detection, TargetShape


class YOLODetector(BaseDetector):
    """Обгортка ultralytics YOLO.

    YOLO віддає лише bbox, тож для кругової цілі еліпс дофітується
    контуром усередині рамки. Без цього PnP і IBVS не мали б форми,
    а сама лише рамка не розрізняє «збоку» і «повернутий».
    """

    supported_shapes = (TargetShape.CIRCLE, TargetShape.RECTANGLE)

    def __init__(
        self,
        weights: str,
        shape: TargetShape = TargetShape.CIRCLE,
        min_confidence: float = 0.25,
        device: str | None = None,
        imgsz: int = 640,
        refine_shape: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(min_confidence=min_confidence)
        from ultralytics import YOLO

        self.weights = weights
        self.shape = shape
        self.imgsz = imgsz
        self.refine_shape = refine_shape
        self.name = name or Path(weights).stem

        self.model = YOLO(weights)
        
        self.device = device or "cpu"
        torch_device = self._as_torch_device(self.device)
        if torch_device is not None:
            self.model.to(torch_device)

    @staticmethod
    def _as_torch_device(device: str) -> str | None:
        d = str(device).strip().lower()
        if d in ("", "cpu"):
            return "cpu"
        if d.isdigit():
            return f"cuda:{d}"
        if d.startswith("cuda"):
            return d
        return None

    def _detect_impl(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.min_confidence,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        detections: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else -1

            det = Detection(
                shape=self.shape,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                meta={"class_id": cls_id, "backend": self.name},
            )

            if self.refine_shape:
                self._refine(frame, det)

            detections.append(det)

        return detections

    def _refine(self, frame: np.ndarray, det: Detection) -> None:
        """Дофітити еліпс/кути всередині рамки, знайденої YOLO.

        Пошук ведеться лише в межах bbox, тому фон майже не заважає —
        це головна перевага гібриду «нейромережа + геометрія»:
        YOLO дає надійне ROI, класична геометрія дає точну форму.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.bbox
        
        pad = 0.06 * max(x2 - x1, y2 - y1)
        rx1 = max(0, int(x1 - pad)); ry1 = max(0, int(y1 - pad))
        rx2 = min(w, int(x2 + pad)); ry2 = min(h, int(y2 + pad))
        if rx2 - rx1 < 8 or ry2 - ry1 < 8:
            return

        roi = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        contour = max(contours, key=cv2.contourArea)

        if det.shape is TargetShape.CIRCLE:
            if len(contour) < 5:
                return
            (cx, cy), axes, angle = cv2.fitEllipse(contour)
            det.ellipse = ((cx + rx1, cy + ry1), axes, angle)
        else:
            perim = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perim, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                from .opencv_detector import order_corners
                corners = approx.reshape(4, 2).astype(np.float64)
                corners[:, 0] += rx1
                corners[:, 1] += ry1
                det.corners = order_corners(corners)


def build_detector(kind: str, shape: TargetShape, device: str | None = None,
                   weights_dir: str = "models") -> BaseDetector:
    """Фабрика для бенчмарку: один виклик на всі три методи."""
    from .opencv_detector import OpenCVDetector

    if kind == "opencv":
        return OpenCVDetector(shape=shape)

    weights_map = {
        "yolov8n": f"{weights_dir}/yolov8n_hoop.pt",
        "yolov11n": f"{weights_dir}/yolo11n_hoop.pt",
    }
    if kind not in weights_map:
        raise ValueError(f"невідомий детектор: {kind}")

    path = weights_map[kind]
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Ваги {path} не знайдено. Спершу натренуйте: "
            f"python -m experiments.train_detectors --data <dataset>/data.yaml"
        )
    return YOLODetector(path, shape=shape, device=device, name=kind)
