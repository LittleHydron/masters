"""Замкнений контур: команди контролера СПРАВДІ рухають камеру.

`make_synthetic_flight.py` рендерить траєкторію, задану наперед: камера
летить за сценарієм незалежно від команд, тому там неможливо провалити
проліт. Це годиться, щоб перевірити осмисленість команд, але не щоб
порівняти стратегії за успішністю.

Тут камера інтегрує власні команди. З'являється те, чого бракувало:
проліт можна не завершити — промазати повз обруч, втратити ціль,
зупинитись. Саме це і є метрика задачі 3.

Модель руху навмисно проста (швидкості без інерції, без затримки):
мета — порівняти стратегії між собою в однакових умовах, а не
відтворити динаміку Tello. Затримку керування додасть окремий
експеримент за задачею новизни.

Приклад:
    python -m experiments.closed_loop_sim --trials 40
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.control.hybrid import HybridController
from src.control.ibvs import IBVSController
from src.control.pbvs import PBVSController
from src.detection.base import Detection, TargetShape
from src.pose.pnp_solver import PnPSolver
from src.sim.dynamics import DroneModel, DroneParams
from src.sim.detector_noise import NoiseParams, apply as apply_noise


def intrinsics(w: int, h: int, fov_deg: float = 82.0) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])


def ring_points(radius: float, normal: np.ndarray, n: int = 220) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 1.0, 0.0])
    if abs(float(normal @ helper)) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)


def project(pts: np.ndarray, K: np.ndarray) -> np.ndarray | None:
    """Спроєктувати точки; None, якщо щось за камерою."""
    if not np.all(pts[:, 2] > 0.20):
        return None
    p = (K @ pts.T).T
    return p[:, :2] / p[:, 2:3]


def detect_local(ring_cam: np.ndarray, K: np.ndarray,
                 w: int, h: int) -> Detection | None:
    """Детекція за точками контуру, вже переведеними в систему камери."""
    uv = project(ring_cam, K)
    if uv is None:
        return None
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    if x2 < 0 or y2 < 0 or x1 > w or y1 > h or (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    ell = cv2.fitEllipse(uv.astype(np.float32))
    return Detection(
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        confidence=0.95, shape=TargetShape.CIRCLE,
        ellipse=((float(ell[0][0]), float(ell[0][1])),
                 (float(ell[1][0]), float(ell[1][1])), float(ell[2])),
    )


def detect(cam: np.ndarray, ring: np.ndarray, K: np.ndarray,
           w: int, h: int) -> Detection | None:
    """Ідеальний детектор: бачить обруч, якщо той у кадрі.

    Детекція тут ідеальна навмисно — порівнюються КОНТРОЛЕРИ, тож
    похибка детектора лише додала б шуму, спільного для всіх трьох.
    """
    uv = project(ring - cam, K)
    if uv is None:
        return None
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    # Ціль має бути хоч частково в кадрі й не виродженою.
    if x2 < 0 or y2 < 0 or x1 > w or y1 > h or (x2 - x1) < 4 or (y2 - y1) < 4:
        return None

    # Еліпс через МНК по спроєктованих точках, а не з габаритів bbox.
    # Нахилений обруч проєктується в НАХИЛЕНИЙ еліпс; якщо підмінити
    # його осесиметричним прямокутником (кут 0, осі = ширина/висота
    # bbox), розклад коніки відновить хибну нормаль — і PBVS керуватиме
    # за сміттям. Саме тут ховалась «непрацездатність» PBVS.
    ell = cv2.fitEllipse(uv.astype(np.float32))
    return Detection(
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        confidence=0.95, shape=TargetShape.CIRCLE,
        ellipse=((float(ell[0][0]), float(ell[0][1])),
                 (float(ell[1][0]), float(ell[1][1])),
                 float(ell[2])),
    )


def run_trial(ctrl, cam0: np.ndarray, normal: np.ndarray, radius: float,
              K: np.ndarray, w: int, h: int, speed: float = 1.5,
              dt: float = 1 / 30, max_steps: int = 2400,
              params: DroneParams | None = None,
              noise: NoiseParams | None = None,
              rng: np.random.Generator | None = None) -> dict:
    """Один проліт. Успіх = перетнути площину обруча всередині кільця."""
    ctrl.reset()
    ring = ring_points(radius, normal)
    rng = rng or np.random.default_rng(0)
    noise = noise or NoiseParams(enabled=False)
    drone = DroneModel(cam0, params or DroneParams(), rng)
    cam = drone.p
    # Знакова відстань до площини обруча. Камера стартує ПЕРЕД обручем,
    # тобто з від'ємного боку (cam.z < 0 при normal=(0,0,1)), і летить
    # у бік додатного. Перетин = зміна знаку з - на +.
    prev_side = float(cam @ normal)
    effort = 0.0
    lost = 0
    prev_cmd = None
    jitter = 0.0

    for step in range(max_steps):
        # Детекція — у системі КАМЕРИ: враховуємо і позицію, і курс.
        # Без повороту обліт неможливий: бічний рух виводив би ціль
        # за межі кадру.
        ring_cam = np.array([drone.rotate_to_camera(pt - drone.p) for pt in ring])
        det = detect_local(ring_cam, K, w, h)
        det = apply_noise(det, float(np.linalg.norm(cam)), rng, noise)
        if det is None:
            lost += 1
        cmd = ctrl.update(det, (h, w), dt)

        prev_cam = cam.copy()
        effort += abs(cmd.lateral) + abs(cmd.forward) + abs(cmd.vertical)
        if prev_cmd is not None:
            jitter += (abs(cmd.lateral - prev_cmd.lateral)
                       + abs(cmd.forward - prev_cmd.forward)
                       + abs(cmd.vertical - prev_cmd.vertical))
        prev_cmd = cmd

        # Динаміка: інерція + затримка замість миттєвого переміщення.
        drone.step(cmd, dt)
        cam = drone.p

        side = float(cam @ normal)
        if prev_side < 0 <= side:
            # Точка перетину: лінійна інтерполяція між кадрами, інакше
            # промах міряється не в площині обруча.
            a = prev_side / (prev_side - side) if prev_side != side else 0.0
            hit = prev_cam + (cam - prev_cam) * a
            radial = float(np.linalg.norm(hit - normal * float(hit @ normal)))
            return {"success": int(radial < radius), "steps": step + 1,
                    "miss_m": round(radial - radius, 3),
                    "effort": round(effort / (step + 1), 4),
                    "jitter": round(jitter / max(1, step), 4),
                    "lost_frames": lost, "outcome": "pass" if radial < radius else "miss"}
        prev_side = side

        if np.linalg.norm(cam) > 12.0:
            return {"success": 0, "steps": step + 1, "miss_m": None,
                    "effort": round(effort / (step + 1), 4),
                    "jitter": round(jitter / max(1, step), 4),
                    "lost_frames": lost, "outcome": "diverged"}

    return {"success": 0, "steps": max_steps, "miss_m": None,
            "effort": round(effort / max_steps, 4),
            "jitter": round(jitter / max(1, max_steps - 1), 4),
            "lost_frames": lost, "outcome": "timeout"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--diameter", type=float, default=0.89)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--switch-distance", type=float, default=2.0)
    ap.add_argument("--speed", type=float, default=1.5,
                    help="макс. швидкість, м/с (Tello ~1.5 у режимі rc)")
    ap.add_argument("--max-steps", type=int, default=2400,
                    help="ліміт кроків на спробу (2400 = 80 с при 30 fps)")
    # Три ефекти вмикаються ОКРЕМО — інакше не видно, який з них
    # що дає. Типово всі увімкнені (реалістичний режим).
    ap.add_argument("--latency", type=float, default=0.20,
                    help="затримка керування, с (0 = вимкнути)")
    ap.add_argument("--max-accel", type=float, default=2.5,
                    help="обмеження прискорення, м/с² (велике = без інерції)")
    ap.add_argument("--noise", action="store_true", default=True,
                    help="шум детектора за виміряною кривою recall")
    ap.add_argument("--no-noise", dest="noise", action="store_false")
    ap.add_argument("--recall-scale", type=float, default=1.0,
                    help="<1 гірший детектор, >1 кращий")
    ap.add_argument("--jitter-px", type=float, default=2.5,
                    help="тремтіння бокса, пікселі")
    ap.add_argument("--ideal", action="store_true",
                    help="вимкнути ВСЕ: ідеальна кінематика й детектор "
                         "(режим старого стенду, для порівняння)")
    ap.add_argument("--out", default="experiments/results_hybrid")
    args = ap.parse_args()

    w, h = args.width, args.height
    K = intrinsics(w, h)
    radius = args.diameter / 2.0
    rng = np.random.default_rng(args.seed)

    # Однакові стартові умови для всіх стратегій — інакше порівняння
    # перетворилось би на порівняння везіння.
    starts = []
    for _ in range(args.trials):
        tilt = np.deg2rad(rng.uniform(-35, 35))
        pan = np.deg2rad(rng.uniform(-25, 25))
        normal = np.array([np.sin(tilt) * np.cos(pan), np.sin(pan), np.cos(tilt) * np.cos(pan)])
        normal /= np.linalg.norm(normal)
        cam0 = np.array([rng.uniform(-1.4, 1.4), rng.uniform(-0.8, 0.8),
                         -rng.uniform(3.0, 5.5)])
        starts.append((cam0, normal))

    def make(kind):
        s = PnPSolver(K, target_diameter=args.diameter)
        if kind == "ibvs":
            return IBVSController()
        if kind == "pbvs":
            return PBVSController(s)
        return HybridController(PBVSController(PnPSolver(K, target_diameter=args.diameter)),
                                IBVSController(), switch_distance=args.switch_distance)

    if args.ideal:
        params = DroneParams(max_speed=args.speed, max_accel=1e6,
                             latency=0.0, drag=0.0, speed_noise=0.0)
        noise = NoiseParams(enabled=False)
    else:
        params = DroneParams(max_speed=args.speed, max_accel=args.max_accel,
                             latency=args.latency)
        noise = NoiseParams(enabled=args.noise, jitter_px=args.jitter_px,
                            recall_scale=args.recall_scale)

    print(f"Режим: {'ІДЕАЛЬНИЙ (стара кінематика)' if args.ideal else 'реалістичний'}"
          f"  затримка={params.latency:.2f}с  приск.={params.max_accel:g} м/с²"
          f"  шум детектора={'так' if noise.enabled else 'ні'}")

    rows = []
    for kind in ("ibvs", "pbvs", "hybrid"):
        for i, (cam0, normal) in enumerate(starts):
            # Свій RNG на спробу: усі стратегії отримують ОДНАКОВІ
            # реалізації шуму, інакше порівняння стане лотереєю.
            r = run_trial(make(kind), cam0, normal, radius, K, w, h,
                          speed=args.speed, max_steps=args.max_steps,
                          params=params, noise=noise,
                          rng=np.random.default_rng(1000 + i))
            r.update({"controller": kind, "trial": i})
            rows.append(r)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "closed_loop.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["controller", "trial", "success",
                                           "outcome", "steps", "miss_m",
                                           "effort", "jitter", "lost_frames"])
        wr.writeheader(); wr.writerows(rows)

    print(f"Спроб на стратегію: {args.trials}\n")
    print(f"{'стратегія':<10}{'успіх':>8}{'проліт':>8}{'промах':>8}"
          f"{'розліт':>8}{'таймаут':>9}{'зусилля':>9}{'смикання':>10}")
    print("=" * 70)
    for kind in ("ibvs", "pbvs", "hybrid"):
        sub = [r for r in rows if r["controller"] == kind]
        n = len(sub)
        oc = {k: sum(1 for r in sub if r["outcome"] == k)
              for k in ("pass", "miss", "diverged", "timeout")}
        print(f"{kind:<10}{sum(r['success'] for r in sub)/n:>7.0%}"
              f"{oc['pass']:>8}{oc['miss']:>8}{oc['diverged']:>8}{oc['timeout']:>9}"
              f"{np.mean([r['effort'] for r in sub]):>9.3f}"
              f"{np.mean([r['jitter'] for r in sub]):>10.4f}")
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
