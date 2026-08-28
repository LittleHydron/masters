"""Згенерувати синтетичний датасет обручів у форматі YOLO.

Обруч із відомою геометрією (за замовчуванням CSP 89 см) рендериться
у випадкових позах на різних фонах. Це прямо адресує залежність від
оточення: модель бачить ту саму ціль у сотнях різних сцен.

Розмітка точна за побудовою; додатково зберігається JSON із позами —
його можна використати для валідації PnP на реальних метричних даних.

Приклад:
    python -m experiments.generate_synthetic_dataset --count 3000
    python -m experiments.generate_synthetic_dataset --count 3000 \
        --backgrounds ~/Pictures/rooms
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synth.hoop_renderer import HoopSpec, render_hoop, look_at_normal
from src.synth.backgrounds import load_real_backgrounds, prepare_background

# Кольори, які реально годяться для цілі (BGR).
COLOR_PRESETS = {
    "red": (40, 40, 200),
    "orange": (30, 110, 240),
    "lime": (60, 220, 90),
    "yellow": (40, 210, 235),
    "blue": (200, 110, 40),
}


def intrinsics(w: int, h: int, fov_deg: float) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])


def photometric_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Змінити освітлення, різкість і шум.

    Це друга половина боротьби з залежністю від середовища: модель
    не повинна чіплятися за конкретну яскравість чи різкість.
    """
    out = img.astype(np.float32)

    # Яскравість і контраст.
    out = out * rng.uniform(0.6, 1.4) + rng.uniform(-30, 30)

    # Кольорова температура: тепла лампа vs денне світло.
    temp = rng.uniform(-25, 25)
    out[:, :, 0] -= temp   # синій
    out[:, :, 2] += temp   # червоний

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Змазування руху — на реальному польоті кадри рідко різкі.
    if rng.random() < 0.35:
        k = int(rng.integers(3, 13)) | 1
        kernel = np.zeros((k, k), np.float32)
        angle = rng.uniform(0, np.pi)
        cx = k // 2
        for i in range(k):
            x = int(round(cx + (i - cx) * np.cos(angle)))
            y = int(round(cx + (i - cx) * np.sin(angle)))
            if 0 <= x < k and 0 <= y < k:
                kernel[y, x] = 1
        s = kernel.sum()
        if s > 0:
            out = cv2.filter2D(out, -1, kernel / s)

    # Розфокус.
    if rng.random() < 0.25:
        k = int(rng.integers(3, 9)) | 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    # Сенсорний шум.
    if rng.random() < 0.5:
        noise = rng.normal(0, rng.uniform(2, 12), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Компресія JPEG — відеопотік Tello стиснений.
    if rng.random() < 0.4:
        q = int(rng.integers(35, 92))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


def sample_pose(rng: np.random.Generator, spec: HoopSpec,
                min_d: float, max_d: float,
                near_ratio: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Випадкова поза обруча, зміщена до типового підльоту."""
    # Ближня зона окремо. Без цього beta(1.6,2.6) дає лише ~6% кадрів
    # ближче 0.8 м, і обруч ПРАКТИЧНО НІКОЛИ не зрізається краєм:
    # у синтетиці v1 `visible_ratio < 1.0` траплявся у 0.0% кадрів.
    # На реальному відео саме такий ракурс (обруч на весь кадр, видно
    # лише дугу) детектор не бачив зовсім — а це момент прольоту,
    # де ціна помилки найвища.
    if rng.random() < near_ratio:
        dist = float(rng.uniform(min_d, min(0.85, max_d)))
    else:
        dist = float(min_d + (max_d - min_d) * rng.beta(1.6, 2.6))
    # Бічне зміщення тим менше, чим ближче — як у реальному підльоті.
    spread = 0.55 * (dist / max_d)
    x = float(rng.normal(0, spread))
    y = float(rng.normal(0, spread * 0.6))
    position = np.array([x, y, dist])

    tilt = float(rng.normal(0, 22))
    pan = float(rng.normal(0, 14))
    tilt = float(np.clip(tilt, -62, 62))
    pan = float(np.clip(pan, -45, 45))
    return position, look_at_normal(tilt, pan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="datasets/synthetic_hoops")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=82.0, help="поле зору камери, град")
    ap.add_argument("--diameter", type=float, default=0.89)
    ap.add_argument("--tube", type=float, default=0.020)
    ap.add_argument("--colors", nargs="+", default=["red", "orange"],
                    choices=list(COLOR_PRESETS))
    ap.add_argument("--backgrounds", default=None,
                    help="папка з фото фонів (якщо немає — процедурні)")
    ap.add_argument("--min-dist", type=float, default=0.9)
    ap.add_argument("--max-dist", type=float, default=6.0)
    ap.add_argument("--near-ratio", type=float, default=0.0,
                    help="частка кадрів у ближній зоні (min-dist..0.85 м), "
                         "де обруч зрізається краєм — критично для "
                         "моменту прольоту")
    ap.add_argument("--empty-ratio", type=float, default=0.08,
                    help="частка кадрів без цілі (негативні приклади)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    K = intrinsics(args.width, args.height, args.fov)
    bg_paths = load_real_backgrounds(args.backgrounds) if args.backgrounds else []
    print(f"Фонів-фотографій: {len(bg_paths) if bg_paths else 0} "
          f"({'реальні' if bg_paths else 'процедурні'})")

    out = Path(args.out)
    for s in ("train", "valid", "test"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)

    tr, va, _ = args.split
    poses: list[dict] = []
    made = 0
    attempts = 0
    max_attempts = args.count * 12

    while made < args.count and attempts < max_attempts:
        attempts += 1
        r = rng.random()
        split = "train" if r < tr else ("valid" if r < tr + va else "test")

        bg_path = bg_paths[int(rng.integers(len(bg_paths)))] if bg_paths else None
        bg = prepare_background(bg_path, args.width, args.height, rng)

        # Негативні приклади: сцена без цілі. Без них модель
        # схильна знаходити обруч там, де його немає.
        if rng.random() < args.empty_ratio:
            img = photometric_aug(bg, rng)
            name = f"synth_{made:06d}"
            cv2.imwrite(str(out / split / "images" / f"{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / split / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
            made += 1
            continue

        color = COLOR_PRESETS[args.colors[int(rng.integers(len(args.colors)))]]
        # Варіація відтінку: реальний обруч не має рівно того кольору.
        color = tuple(int(np.clip(c + rng.normal(0, 16), 0, 255)) for c in color)
        spec = HoopSpec(diameter=args.diameter, tube_diameter=args.tube,
                        color_bgr=color)

        position, normal = sample_pose(rng, spec, args.min_dist, args.max_dist,
                                       args.near_ratio)
        res = render_hoop(bg, K, position, normal, spec, rng=rng)
        if res is None or res.visible_ratio < 0.3:
            continue

        img = photometric_aug(res.image, rng)

        x1, y1, x2, y2 = res.bbox
        cx = (x1 + x2) / 2.0 / args.width
        cy = (y1 + y2) / 2.0 / args.height
        bw = (x2 - x1) / args.width
        bh = (y2 - y1) / args.height
        if bw <= 0.004 or bh <= 0.004:
            continue

        name = f"synth_{made:06d}"
        cv2.imwrite(str(out / split / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / split / "labels" / f"{name}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

        poses.append({
            "name": name, "split": split,
            "position": [round(float(v), 4) for v in res.position],
            "normal": [round(float(v), 4) for v in res.normal],
            "distance": round(float(np.linalg.norm(res.position)), 4),
            "visible_ratio": round(res.visible_ratio, 3),
            "ellipse": [[round(float(v), 2) for v in res.ellipse[0]],
                        [round(float(v), 2) for v in res.ellipse[1]],
                        round(float(res.ellipse[2]), 2)] if res.ellipse else None,
        })
        made += 1
        if made % 500 == 0:
            print(f"  згенеровано {made}/{args.count}")

    root = out.resolve()
    (out / "data.yaml").write_text(
        "# Синтетичний датасет обручів.\n"
        f"# Обруч: діаметр {args.diameter} м, трубка {args.tube} м.\n"
        f"# Камера: {args.width}x{args.height}, поле зору {args.fov} град.\n"
        "# Розмітка точна за побудовою; пози збережено в poses.json\n"
        "# і придатні для валідації оцінки 3D-пози.\n"
        f"path: {root}\ntrain: train/images\nval: valid/images\ntest: test/images\n\n"
        "nc: 1\nnames: ['ring']\n", encoding="utf-8")
    (out / "poses.json").write_text(
        json.dumps(poses, indent=1, ensure_ascii=False), encoding="utf-8")

    counts = {s: len(list((out / s / "images").iterdir())) for s in ("train", "valid", "test")}
    print(f"\nЗгенеровано {made} зображень (спроб {attempts})")
    print(f"  {counts}")
    print(f"  датасет: {out / 'data.yaml'}")
    print(f"  пози:    {out / 'poses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
