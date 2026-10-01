"""Крос-датасетна перевірка: чи узагальнюються натреновані моделі.

Моделі, натреновані на drone-racing-okmii, прогоняються на ІНШИХ
датасетах кільцеподібних об'єктів. Різке падіння метрик означає,
що висока точність на рідному тесті пояснюється витоком даних
(аугментації одного кадру потрапили і в train, і в test), а не
якістю детекції.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.benchmark_detectors import iou, load_gt


def eval_on(weights: str, img_dir: Path, lab_dir: Path, device: str,
            conf: float = 0.25, iou_thr: float = 0.5) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in exts)

    tp = fp = fn = 0
    confs_hit: list[float] = []
    for p in images:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        res = model.predict(frame, conf=conf, verbose=False, device=device)
        dets = []
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                dets.append(((x1, y1, x2, y2), float(b.conf[0])))

        gt = load_gt(lab_dir / (p.stem + ".txt"), w, h)
        matched = set()
        for box, c in sorted(dets, key=lambda d: -d[1]):
            best_i, best_v = -1, 0.0
            for i, g in enumerate(gt):
                if i in matched:
                    continue
                v = iou(box, g)
                if v > best_v:
                    best_i, best_v = i, v
            if best_v >= iou_thr:
                tp += 1
                matched.add(best_i)
                confs_hit.append(c)
            else:
                fp += 1
        fn += len(gt) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "images": len(images), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_conf_hit": round(float(np.mean(confs_hit)), 4) if confs_hit else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--tags", nargs="+", default=["hoop"],
                    help="які варіанти навчання порівнювати: hoop full dedup mixed")
    ap.add_argument("--name", default="cross_dataset_eval",
                    help="ім'я вихідного JSON без розширення")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    targets = [
        ("рідний тест (okmii)", Path("datasets/hoops/test")),
        ("synth_holdout", Path("datasets/synthetic_holdout/test")),
        ("notedetection", Path("datasets/xval/notedetection/test")),
        ("note_detection", Path("datasets/xval/note_detection/test")),
    ]
    models = {}
    for tag in args.tags:
        for key, stem in (("yolov8n", "yolov8n"), ("yolov11n", "yolo11n")):
            models[f"{key}_{tag}"] = Path(args.models_dir) / f"{stem}_{tag}.pt"

    rows = []
    for mname, wpath in models.items():
        if not wpath.exists():
            print(f"пропуск {mname}: немає {wpath}")
            continue
        for tname, troot in targets:
            img_dir, lab_dir = troot / "images", troot / "labels"
            if not img_dir.exists():
                print(f"пропуск {tname}: немає {img_dir}")
                continue
            r = eval_on(str(wpath), img_dir, lab_dir, args.device)
            r["model"], r["dataset"] = mname, tname
            rows.append(r)
            print(f"{mname:17s} на {tname:22s} "
                  f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
                  f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 81}")
    print(f"{'модель':<18}{'датасет':<24}{'P':>8}{'R':>8}{'F1':>8}{'детекцій':>10}")
    print("=" * 81)
    for r in rows:
        print(f"{r['model']:<18}{r['dataset']:<24}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['tp'] + r['fp']:>10}")
    print(f"\nЗбережено: {out / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
