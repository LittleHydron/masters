"""Витягти рівномірну вибірку кадрів із реальних відео для розмітки.

Без розмітки ми міряємо лише ЧАСТКУ КАДРІВ ІЗ ДЕТЕКЦІЄЮ — тобто «чи
модель щось знайшла». Кадр із боксом на віконній рамі рахується як
успіх. Розмітка дає справжні precision/recall/F1 і робить результати
зіставними з тестовими наборами.

Вибірка рівномірна за часом: беремо кожен N-й кадр так, щоб набрати
задану кількість. Це чесніше за випадкову — покриває весь політ,
включно з ближньою зоною й поворотами.

Приклад:
    python -m experiments.make_label_set --per-video 35
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+",
                    default=["Kitchen", "Balkony", "Outside", "Room"])
    ap.add_argument("--src", default="datasets/real")
    ap.add_argument("--per-video", type=int, default=35)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="зменшити кадр (розмічати зручніше, точність та сама)")
    ap.add_argument("--out", default="datasets/real_labeled")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    index = []
    for name in args.videos:
        path = Path(args.src) / f"{name}.mov"
        if not path.exists():
            print(f"пропуск {name}: немає {path}")
            continue
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // args.per_video)
        taken = 0
        idx = 0
        while taken < args.per_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            if args.scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale)
            fname = f"{name}_{idx:06d}.jpg"
            cv2.imwrite(str(out / "images" / fname), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            index.append({"file": fname, "video": name, "frame": idx})
            taken += 1
            idx += step
            if idx >= total:
                break
        cap.release()
        print(f"  {name}: {taken} кадрів (крок {step} з {total})")

    (out / "index.json").write_text(
        json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "classes.txt").write_text("ring\n", encoding="utf-8")
    print(f"\nУсього: {len(index)} кадрів -> {out / 'images'}")
    print(f"Розмітка йтиме у {out / 'labels'} (формат YOLO)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
