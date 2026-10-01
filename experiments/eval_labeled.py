"""Оцінити всі моделі на РОЗМІЧЕНИХ реальних кадрах.

До цього ми міряли лише «частку кадрів із детекцією» — тобто чи модель
щось знайшла. Кадр із боксом на віконній рамі рахувався як успіх.
Тут є ground truth, тож рахуються справжні precision / recall / F1
із порогом IoU.

Метрики окремо ПО СЦЕНАХ: кадри в межах одного відео сильно
корельовані (та сама кімната, той самий обруч), тож 140 кадрів — це
не 140 незалежних вимірювань, а радше 4 спостереження. Одне середнє
число приховувало б розкид.

Приклад:
    python -m experiments.eval_labeled
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark_detectors import iou, load_gt


def evaluate(weights: str, images: list[Path], lbl_dir: Path,
             conf: float, iou_thr: float, device: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(weights)
    per_scene: dict[str, dict] = {}

    for img_path in images:
        scene = img_path.stem.split("_")[0]
        st = per_scene.setdefault(scene, {"tp": 0, "fp": 0, "fn": 0,
                                          "ious": [], "confs": []})
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = load_gt(lbl_dir / (img_path.stem + ".txt"), w, h)

        res = model.predict(frame, conf=conf, verbose=False, device=device)
        boxes = res[0].boxes
        dets = []
        if boxes is not None and len(boxes):
            for b in boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                dets.append(((x1, y1, x2, y2), float(b.conf[0])))
        dets.sort(key=lambda d: -d[1])

        matched: set[int] = set()
        for box, c in dets:
            best_i, best_v = -1, 0.0
            for i, g in enumerate(gt):
                if i in matched:
                    continue
                v = iou(box, g)
                if v > best_v:
                    best_i, best_v = i, v
            if best_v >= iou_thr:
                st["tp"] += 1
                matched.add(best_i)
                st["ious"].append(best_v)
                st["confs"].append(c)
            else:
                st["fp"] += 1
        st["fn"] += len(gt) - len(matched)

    out = {}
    for scene, st in per_scene.items():
        p = st["tp"] / (st["tp"] + st["fp"]) if (st["tp"] + st["fp"]) else 0.0
        r = st["tp"] / (st["tp"] + st["fn"]) if (st["tp"] + st["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[scene] = {
            "tp": st["tp"], "fp": st["fp"], "fn": st["fn"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "mean_iou": round(float(np.mean(st["ious"])), 4) if st["ious"] else 0.0,
            "median_iou": round(float(np.median(st["ious"])), 4) if st["ious"] else 0.0,
            "mean_conf": round(float(np.mean(st["confs"])), 4) if st["confs"] else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/real_labeled")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="experiments/results/labeled_eval.json")
    args = ap.parse_args()

    root = Path(args.root)
    images = sorted((root / "images").glob("*.jpg"))
    lbl_dir = root / "labels"
    if not images:
        raise SystemExit(f"Немає зображень у {root/'images'}")

    md = Path(args.models_dir)
    models = [p for p in sorted(md.glob("*.pt"))
              if "b16" not in p.stem and "hoop" not in p.stem]

    results = {}
    for m in models:
        print(f"  {m.stem} ...", flush=True)
        results[m.stem] = evaluate(str(m), images, lbl_dir,
                                   args.conf, args.iou, args.device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    def fbeta(p: float, r: float, beta: float) -> float:
        """F-міра з вагою precision. beta<1 підсилює precision.

        Для дрона помилки НЕ рівноцінні: пропуск означає кадр без цілі
        (контролер тримає курс, наступний кадр за 33 мс), а хибна
        детекція веде дрон на сторонній об'єкт. Тому beta=0.5.
        """
        if p + r <= 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)

    scenes = ["Kitchen", "Balkony", "Outside", "Room"]
    rows = []
    for name, per in results.items():
        tp = sum(per[s]["tp"] for s in scenes if s in per)
        fp = sum(per[s]["fp"] for s in scenes if s in per)
        fn = sum(per[s]["fn"] for s in scenes if s in per)
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        med = [per[s]["median_iou"] for s in scenes
               if s in per and per[s]["median_iou"]]
        mean = [per[s]["mean_iou"] for s in scenes
                if s in per and per[s]["mean_iou"]]
        rows.append({
            "name": name, "p": p_, "r": r_,
            "f1": fbeta(p_, r_, 1.0), "f05": fbeta(p_, r_, 0.5),
            "med_iou": float(np.median(med)) if med else 0.0,
            "mean_iou": float(np.mean(mean)) if mean else 0.0,
            "per_scene_f05": [fbeta(per[s]["precision"], per[s]["recall"], 0.5)
                              if s in per else 0.0 for s in scenes],
        })
    rows.sort(key=lambda x: -x["f05"])

    print(f"\n{'модель':<20}{'P':>7}{'R':>7}{'F1':>7}{'F0.5':>8}"
          f"{'IoU_med':>9}{'IoU_avg':>9}")
    print("=" * 68)
    for x in rows:
        print(f"{x['name']:<20}{x['p']:>7.3f}{x['r']:>7.3f}{x['f1']:>7.3f}"
              f"{x['f05']:>8.3f}{x['med_iou']:>9.3f}{x['mean_iou']:>9.3f}")

    print(f"\nF0.5 по сценах:")
    print(f"{'модель':<20}" + "".join(f"{s[:8]:>10}" for s in scenes))
    print("=" * 60)
    for x in rows:
        print(f"{x['name']:<20}" + "".join(f"{v:>10.3f}" for v in x["per_scene_f05"]))
    print(f"\nЗбережено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
