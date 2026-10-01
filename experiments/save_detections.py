"""Зберегти детекції покадрово, щоб результат можна було перевірити очима.

Одних лише зведених відсотків недостатньо: щоб довіряти цифрі, треба
бачити кадри, на яких вона отримана. Скрипт зберігає:

* `frames/` — кадри з накладеною детекцією (кожен N-й);
* `misses/` — ОКРЕМО кадри, де детекції не було: саме там ховаються
  проблеми, і саме їх найкорисніше переглядати;
* `contact_sheet.jpg` — оглядова сітка;
* `detections.csv` — покадрові числа.

Приклад:
    python -m experiments.save_detections --video datasets/real/Room.mov \\
        --weights models/yolov8n_mixed.pt --every 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.track_select import StickyTrackSelector


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="models/yolov8n_mixed.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=10, help="зберігати кожен N-й кадр")
    ap.add_argument("--device", default="0")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--track", action="store_true",
                    help="увімкнути ByteTrack зі sticky-вибором треку: "
                         "прибирає перескоки бокса на сторонні об'єкти")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--out", default="experiments/detections")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    tag = Path(args.weights).stem + ("_track" if args.track else "")
    selector = StickyTrackSelector()
    stem = Path(args.video).stem
    root = Path(args.out) / f"{stem}__{tag}"
    (root / "frames").mkdir(parents=True, exist_ok=True)
    (root / "misses").mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Не відкрити {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows, thumbs = [], []
    idx = saved = hits = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        keep = (idx % args.every == 0)
        if not args.track and not keep:
            continue

        if args.track:
            res = model.track(frame, persist=True, tracker=args.tracker,
                              conf=args.conf, verbose=False, device=args.device)
        else:
            res = model.predict(frame, conf=args.conf, verbose=False,
                                device=args.device)
        boxes = res[0].boxes
        found = boxes is not None and len(boxes) > 0
        conf = 0.0
        k = 0
        if found:
            ids = ([int(t) for t in boxes.id.tolist()]
                   if args.track and boxes.id is not None else None)
            k = selector.select(ids, [float(c) for c in boxes.conf])
            conf = float(boxes.conf[k])
        if not keep:
            continue
        saved += 1
        if found:
            hits += 1
        vis = frame.copy()
        if found:
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[k].tolist()]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 0), 3)
            cv2.putText(vis, f"{conf:.2f}", (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 0), 2)
        else:
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1),
                          (0, 0, 235), 8)
            cv2.putText(vis, "NO DETECTION", (24, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 235), 3)
        cv2.putText(vis, f"frame {idx}", (24, vis.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        small = cv2.resize(vis, (0, 0), fx=args.scale, fy=args.scale)
        cv2.imwrite(str(root / "frames" / f"{idx:06d}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not found:
            cv2.imwrite(str(root / "misses" / f"{idx:06d}.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        if len(thumbs) < 40:
            thumbs.append(cv2.resize(vis, (320, 180)))
        rows.append({"frame": idx, "detected": int(found),
                     "conf": round(conf, 4)})

    cap.release()

    if thumbs:
        cols = 5
        while len(thumbs) % cols:
            thumbs.append(np.zeros((180, 320, 3), np.uint8))
        grid = np.vstack([np.hstack(thumbs[i:i + cols])
                          for i in range(0, len(thumbs), cols)])
        cv2.imwrite(str(root / "contact_sheet.jpg"), grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

    with open(root / "detections.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "detected", "conf"])
        wr.writeheader()
        wr.writerows(rows)

    rate = hits / max(1, saved)
    (root / "README.txt").write_text(
        f"Відео:    {args.video}\n"
        f"Модель:   {args.weights}\n"
        f"Поріг:    conf={args.conf}\n"
        f"Трекінг:  {'ByteTrack + sticky' if args.track else 'немає (predict)'}\n"
        f"Кадрів у відео: {total}, перевірено кожен {args.every}-й = {saved}\n"
        f"Детекцій: {hits}/{saved} = {rate:.1%}\n\n"
        f"frames/         — усі перевірені кадри з рамкою детекції\n"
        f"misses/         — ЛИШЕ кадри без детекції ({saved - hits} шт.)\n"
        f"contact_sheet.jpg — оглядова сітка перших 40 кадрів\n"
        f"detections.csv  — покадрово: frame, detected, conf\n",
        encoding="utf-8")

    print(f"{stem} / {tag}: {hits}/{saved} = {rate:.1%}"
          + (f"  перемикань треку: {selector.switches}" if args.track else ""))
    print(f"  кадри:   {root/'frames'}")
    print(f"  промахи: {root/'misses'} ({saved-hits} шт.)")
    print(f"  сітка:   {root/'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
