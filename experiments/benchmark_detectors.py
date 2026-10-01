"""Порівняння детекторів за точністю та швидкодією (задача 1).

Міряє на ОДНОМУ тестовому наборі:
  - OpenCV (лише CPU — інакше не працює)
  - YOLOv8n  на GPU і на CPU
  - YOLOv11n на GPU і на CPU

Заміри CPU потрібні для чесного порівняння з OpenCV на однаковому
залізі; заміри GPU показують, що реально буде на демо з ноута.

Приклад:
    python -m experiments.benchmark_detectors --data datasets/hoops/data.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.base import TargetShape, Detection


def load_test_images(data_yaml: Path) -> tuple[list[Path], Path | None]:
    """Дістати список тестових зображень із data.yaml Roboflow."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent
    base = Path(cfg.get("path", ".")) if cfg.get("path") else Path(".")
    if not base.is_absolute():
        base = (root / base).resolve()

    split = cfg.get("test") or cfg.get("val")
    if split is None:
        raise SystemExit("У data.yaml немає ні test, ні val")

    img_dir = (base / split).resolve()
    if not img_dir.exists():
        img_dir = (root / split).resolve()
    if img_dir.is_file():
        img_dir = img_dir.parent

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise SystemExit(f"Не знайдено зображень у {img_dir}")

    label_dir = Path(str(img_dir).replace("/images", "/labels"))
    return images, (label_dir if label_dir.exists() else None)


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Прочитати YOLO-розмітку -> список bbox у пікселях."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        boxes.append((x1 * w, y1 * h, x2 * w, y2 * h))
    return boxes


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(detector, images: list[Path], label_dir: Path | None,
             iou_thr: float = 0.5, warmup: int = 5) -> dict:
    """Прогнати детектор і зібрати точність + час інференсу."""
    if images:
        probe = cv2.imread(str(images[0]))
        if probe is not None:
            for _ in range(warmup):
                detector.detect(probe)

    tp = fp = fn = 0
    times: list[float] = []
    detected_frames = 0

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        dets = detector.detect(frame)
        times.append((time.perf_counter() - t0) * 1000.0)

        if dets:
            detected_frames += 1

        if label_dir is None:
            continue

        gt = load_gt(label_dir / (img_path.stem + ".txt"), w, h)
        matched = set()
        for det in sorted(dets, key=lambda d: -d.confidence):
            best_i, best_iou = -1, 0.0
            for i, g in enumerate(gt):
                if i in matched:
                    continue
                v = iou(det.bbox, g)
                if v > best_iou:
                    best_i, best_iou = i, v
            if best_iou >= iou_thr:
                tp += 1
                matched.add(best_i)
            else:
                fp += 1
        fn += len(gt) - len(matched)

    times_arr = np.array(times) if times else np.array([0.0])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "images": len(times),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "detection_rate": round(detected_frames / max(1, len(times)), 4),
        "mean_ms": round(float(times_arr.mean()), 2),
        "median_ms": round(float(np.median(times_arr)), 2),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 2),
        "fps": round(1000.0 / float(times_arr.mean()), 1) if times_arr.mean() > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="шлях до data.yaml")
    ap.add_argument("--shape", choices=["circle", "rectangle"], default="circle")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="обмежити кількість зображень")
    ap.add_argument("--skip-cpu", action="store_true",
                    help="не міряти YOLO на CPU (довго)")
    args = ap.parse_args()

    shape = TargetShape.CIRCLE if args.shape == "circle" else TargetShape.RECTANGLE
    images, label_dir = load_test_images(Path(args.data))
    if args.limit:
        images = images[:args.limit]

    print(f"Тестових зображень: {len(images)}")
    print(f"Розмітка: {label_dir if label_dir else 'НЕМАЄ (тільки швидкодія)'}\n")

    import torch
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "—"
    print(f"GPU: {gpu_name}" if has_cuda else "GPU недоступний, лише CPU")

    from src.detection.opencv_detector import OpenCVDetector
    from src.detection.yolo_detector import YOLODetector

    configs: list[tuple[str, str, object]] = [
        ("opencv_contour", "cpu", OpenCVDetector(shape=shape)),
    ]

    weights = {
        "yolov8n": Path(args.models_dir) / "yolov8n_hoop.pt",
        "yolov11n": Path(args.models_dir) / "yolo11n_hoop.pt",
    }
    for key, wpath in weights.items():
        if not wpath.exists():
            print(f"  пропуск {key}: немає {wpath}")
            continue
        if has_cuda:
            configs.append((key, "cuda",
                            YOLODetector(str(wpath), shape=shape, device="0", name=key)))
        if not args.skip_cpu:
            configs.append((key, "cpu",
                            YOLODetector(str(wpath), shape=shape, device="cpu", name=key)))

    if len(configs) == 1:
        print("\nЗнайдено лише OpenCV. Спершу натренуйте моделі:\n"
              "  python -m experiments.train_detectors --data <data.yaml>")

    rows = []
    for name, device, detector in configs:
        print(f"\n--- {name} [{device}] ---")
        res = evaluate(detector, images, label_dir, iou_thr=args.iou)
        res["detector"] = name
        res["device"] = device
        rows.append(res)
        print(f"    P={res['precision']:.3f} R={res['recall']:.3f} "
              f"F1={res['f1']:.3f} | {res['mean_ms']:.1f} мс "
              f"({res['fps']:.1f} FPS)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = ["detector", "device", "images", "tp", "fp", "fn", "precision",
              "recall", "f1", "detection_rate", "mean_ms", "median_ms",
              "p95_ms", "fps"]
    csv_path = out_dir / "detector_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows([{k: r.get(k) for k in fields} for r in rows])

    (out_dir / "detector_benchmark.json").write_text(
        json.dumps({"gpu": gpu_name, "results": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(f"\n{'=' * 76}")
    print(f"{'детектор':<18}{'пристрій':<10}{'P':>8}{'R':>8}{'F1':>8}"
          f"{'мс':>9}{'FPS':>8}")
    print("=" * 76)
    for r in rows:
        print(f"{r['detector']:<18}{r['device']:<10}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['mean_ms']:>9.1f}{r['fps']:>8.1f}")
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
