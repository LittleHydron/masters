"""Зняття дзеркальної двозначності пози за рухом камери.

Проєкція кола є еліпсом, і розклад коніки дає ДВА валідних розв'язки
з однаковим центром, але дзеркальними нормалями. З одного кадру вони
нерозрізненні.

Ідея перевірки: якщо камера зсунулась на відомий вектор, кожна гілка
ПРОГНОЗУЄ свій еліпс на наступному кадрі. Правильна гілка дає меншу
нев'язку з фактично виміряним еліпсом.

Важливо: рух УЗДОВЖ осі обруча не допомагає — еліпс зростає масштабно,
і гілки лишаються нерозрізненними. Інформацію дає БІЧНИЙ зсув.

Другий кадр рендериться тим самим кодом, що й датасет (`render_hoop`),
тому тест перевіряє метод, а не рендерер.

Приклад:
    python -m experiments.disambiguate_by_motion --shift 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pose.pnp_solver import PnPSolver
from src.synth.hoop_renderer import HoopSpec, render_hoop


def intrinsics(w: int, h: int, fov_deg: float = 82.0) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])


def ellipse_from_pose(position: np.ndarray, normal: np.ndarray,
                      radius: float, K: np.ndarray) -> np.ndarray | None:
    """Спроєктувати коло заданої пози -> параметри еліпса (5 чисел).

    Повертає (cx, cy, a, b, angle_rad) або None, якщо не проєктується.
    """
    n = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(n @ helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, helper); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    pts = position + radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    if not np.all(pts[:, 2] > 0.05):
        return None
    p = (K @ pts.T).T
    uv = (p[:, :2] / p[:, 2:3]).astype(np.float32)
    (cx, cy), (aw, ah), ang = cv2.fitEllipse(uv)
    return np.array([cx, cy, aw, ah, np.deg2rad(ang)])


def residual(pred: np.ndarray, obs: np.ndarray) -> float:
    """Нев'язка між прогнозованим і виміряним еліпсом.

    Центр і півосі — у пікселях. Кут окремо: він циклічний з періодом
    pi (еліпс симетричний), тому різницю згортаємо у [-pi/2, pi/2] і
    зважуємо характерним розміром цілі, щоб градуси й пікселі були
    зіставні.
    """
    d_center = np.linalg.norm(pred[:2] - obs[:2])
    d_axes = abs(pred[2] - obs[2]) + abs(pred[3] - obs[3])
    da = (pred[4] - obs[4]) % np.pi
    if da > np.pi / 2:
        da -= np.pi
    scale = 0.5 * (obs[2] + obs[3])
    return float(d_center + d_axes + abs(da) * scale)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", default="datasets/synthetic_holdout/poses.json")
    ap.add_argument("--diameter", type=float, default=1.20,
                    help="діаметр обруча в holdout")
    ap.add_argument("--tube", type=float, default=0.045)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--shift", type=float, default=0.15,
                    help="бічний зсув камери між кадрами, м")
    ap.add_argument("--axial", action="store_true",
                    help="зсув УЗДОВЖ осі огляду замість бічного "
                         "(контроль: метод НЕ має працювати)")
    ap.add_argument("--noise-px", type=float, default=0.0,
                    help="шум вимірювання еліпса, пікселі (реальний "
                         "детектор ідеального еліпса не дає)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="experiments/results/disambiguation.json")
    args = ap.parse_args()

    w, h = args.width, args.height
    K = intrinsics(w, h)
    radius = args.diameter / 2.0
    solver = PnPSolver(K, target_diameter=args.diameter)
    spec = HoopSpec(diameter=args.diameter, tube_diameter=args.tube)
    rng = np.random.default_rng(0)

    poses = json.loads(Path(args.poses).read_text(encoding="utf-8"))
    if args.limit:
        poses = poses[:args.limit]

    stats = {"total": 0, "no_second_solution": 0, "render_failed": 0,
             "correct": 0, "wrong": 0, "tie": 0}
    margins = []
    per_tilt = {}

    for rec in poses:
        pos = np.array(rec["position"], dtype=float)
        n_true = np.array(rec["normal"], dtype=float)
        n_true /= np.linalg.norm(n_true)

        e1 = ellipse_from_pose(pos, n_true, radius, K)
        if e1 is None:
            stats["render_failed"] += 1
            continue
        if args.noise_px > 0:
            e1 = e1.copy()
            e1[:2] += rng.normal(0, args.noise_px, 2)
            e1[2:4] += rng.normal(0, args.noise_px, 2)
            e1[4] += rng.normal(0, np.deg2rad(args.noise_px))

        from src.detection.base import Detection, TargetShape
        det = Detection(
            bbox=(float(e1[0] - e1[2] / 2), float(e1[1] - e1[3] / 2),
                  float(e1[0] + e1[2] / 2), float(e1[1] + e1[3] / 2)),
            confidence=1.0, shape=TargetShape.CIRCLE,
            ellipse=((float(e1[0]), float(e1[1])),
                     (float(e1[2]), float(e1[3])), float(np.degrees(e1[4]))),
        )
        est = solver.solve(det)
        if est is None or est.alternative is None:
            stats["no_second_solution"] += 1
            continue

        branches = [est, est.alternative]
        stats["total"] += 1

        if args.axial:
            shift = np.array([0.0, 0.0, args.shift])
        else:
            shift = np.array([args.shift, 0.0, 0.0])

        pos2_true = pos - shift
        e2_obs = ellipse_from_pose(pos2_true, n_true, radius, K)
        if e2_obs is not None and args.noise_px > 0:
            e2_obs = e2_obs.copy()
            e2_obs[:2] += rng.normal(0, args.noise_px, 2)
            e2_obs[2:4] += rng.normal(0, args.noise_px, 2)
            e2_obs[4] += rng.normal(0, np.deg2rad(args.noise_px))
        if e2_obs is None:
            stats["render_failed"] += 1
            stats["total"] -= 1
            continue

        res = []
        for b in branches:
            p2 = b.position - shift
            e2_pred = ellipse_from_pose(p2, b.normal, radius, K)
            res.append(np.inf if e2_pred is None else residual(e2_pred, e2_obs))

        align = [abs(float(b.normal @ n_true)) for b in branches]
        truth_idx = int(np.argmax(align))

        if not np.isfinite(res[0]) and not np.isfinite(res[1]):
            stats["render_failed"] += 1
            stats["total"] -= 1
            continue

        pick = int(np.argmin(res))
        if abs(align[0] - align[1]) < 1e-3:
            stats["tie"] += 1
            continue

        if pick == truth_idx:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1

        lo, hi = sorted(res)
        margins.append(0.0 if not np.isfinite(hi) else (hi - lo) / max(1e-9, hi))

        tilt = np.degrees(np.arccos(np.clip(abs(n_true[2]), 0, 1)))
        key = f"{int(tilt // 10) * 10}-{int(tilt // 10) * 10 + 10}"
        d = per_tilt.setdefault(key, {"correct": 0, "wrong": 0})
        d["correct" if pick == truth_idx else "wrong"] += 1

    decided = stats["correct"] + stats["wrong"]
    acc = stats["correct"] / decided if decided else 0.0

    print(f"Зсув: {'уздовж осі (контроль)' if args.axial else 'бічний'} "
          f"{args.shift} м")
    print(f"Кадрів придатних: {stats['total']}")
    print(f"  визначено: {decided}, нерозрізненні (майже фронтальні): {stats['tie']}")
    print(f"  ПРАВИЛЬНО: {stats['correct']}  помилково: {stats['wrong']}")
    print(f"  точність: {acc:.1%}")
    if margins:
        print(f"  медіанний відрив нев'язок: {np.median(margins):.1%}")

    if per_tilt:
        print("\nЗа нахилом обруча:")
        for k in sorted(per_tilt, key=lambda s: int(s.split('-')[0])):
            v = per_tilt[k]
            n = v["correct"] + v["wrong"]
            print(f"  {k:>7}°: {v['correct']/n:6.1%}  (n={n})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"shift": args.shift, "axial": bool(args.axial), "stats": stats,
         "accuracy": round(acc, 4), "per_tilt": per_tilt},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nЗбережено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
