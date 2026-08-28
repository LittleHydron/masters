"""Конвертувати розмітку labelme (JSON) у формат YOLO (.txt).

labelme зберігає анотації у власний JSON із полігонами або
прямокутниками. Скрипти проєкту читають YOLO: один рядок на об'єкт,
`клас cx cy w h` у частках від розміру кадру.

Кадри БЕЗ об'єкта дають порожній .txt — це валідна розмітка, потрібна
щоб рахувати хибні спрацювання. Тому файл створюється для кожного
зображення, навіть якщо JSON порожній або відсутній.

Приклад:
    python -m experiments.labelme_to_yolo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/real_labeled")
    ap.add_argument("--json-dir", default=None,
                    help="де лежать JSON labelme (типово images/)")
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "images"
    json_dir = Path(args.json_dir) if args.json_dir else img_dir
    lbl_dir = root / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"Немає зображень у {img_dir}")

    with_box = empty = missing = 0
    for img in images:
        js = json_dir / (img.stem + ".json")
        out = lbl_dir / (img.stem + ".txt")

        if not js.exists():
            missing += 1
            continue

        data = json.loads(js.read_text(encoding="utf-8"))
        w = float(data.get("imageWidth") or 0)
        h = float(data.get("imageHeight") or 0)
        if w <= 0 or h <= 0:
            missing += 1
            continue

        lines = []
        for sh in data.get("shapes", []):
            pts = sh.get("points") or []
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # Працює і для rectangle (2 точки), і для полігона (N точок):
            # беремо обгортковий прямокутник.
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            if bw <= 0.001 or bh <= 0.001:
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out.write_text("\n".join(lines) + ("\n" if lines else ""),
                       encoding="utf-8")
        if lines:
            with_box += 1
        else:
            empty += 1

    print(f"Зображень: {len(images)}")
    print(f"  з рамкою:        {with_box}")
    print(f"  порожніх (без цілі): {empty}")
    if missing:
        print(f"  БЕЗ JSON (не розмічено): {missing}")
    print(f"Розмітка YOLO: {lbl_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
