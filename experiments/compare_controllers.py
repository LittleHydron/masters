"""Паралельний прогін IBVS і PBVS на однакових кадрах.

Джерело кадрів: відеофайл, вебкамера або телефон (IP-камера).
Дрон не потрібен — команди не надсилаються, а записуються й
візуалізуються. Це і є критерій вибору стратегії керування.

Приклади:
    python -m experiments.compare_controllers --source video.mp4
    python -m experiments.compare_controllers --source 0
    python -m experiments.compare_controllers --source http://192.168.0.5:8080/video
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.base import TargetShape
from src.detection.opencv_detector import OpenCVDetector
from src.pose.pnp_solver import PnPSolver
from src.control.ibvs import IBVSController
from src.control.pbvs import PBVSController
from src.control.hybrid import HybridController
from src.viz.overlay import draw_comparison


def default_intrinsics(w: int, h: int) -> np.ndarray:
    """Груба інтринсика, якщо калібрування ще нема.

    Припущення: горизонтальне поле зору ~82 град (як у Tello).
    Для IBVS це неважливо; для PBVS дає систематичну похибку —
    і саме цю різницю в чутливості ми й хочемо побачити.
    """
    fov_x = np.deg2rad(82.0)
    fx = (w / 2.0) / np.tan(fov_x / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def open_source(source: str) -> cv2.VideoCapture:
    """Відкрити відеофайл, камеру за індексом або мережевий потік."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Не вдалося відкрити джерело: {source}")
    return cap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="відеофайл, індекс камери (0) або URL потоку")
    ap.add_argument("--shape", choices=["circle", "rectangle"], default="circle")
    ap.add_argument("--switch-distance", type=float, default=2.0,
                    help="дистанція перемикання PBVS->IBVS у гібриді, м")
    ap.add_argument("--diameter", type=float, default=0.60,
                    help="реальний діаметр обруча, м")
    ap.add_argument("--target-size", type=float, nargs=2, default=[0.60, 0.60],
                    metavar=("W", "H"), help="реальний розмір рамки, м")
    ap.add_argument("--out", default="experiments/results",
                    help="куди писати CSV і відео")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--no-display", action="store_true",
                    help="не відкривати вікно (для запуску без GUI)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    shape = TargetShape.CIRCLE if args.shape == "circle" else TargetShape.RECTANGLE
    cap = open_source(args.source)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Джерело не віддало жодного кадру")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    dt = 1.0 / fps

    K = default_intrinsics(w, h)
    detector = OpenCVDetector(shape=shape)
    solver = PnPSolver(K, target_diameter=args.diameter,
                       target_size=tuple(args.target_size))

    ibvs = IBVSController()
    pbvs = PBVSController(solver)
    hybrid = HybridController(
        PBVSController(PnPSolver(K, target_diameter=args.diameter,
                                 target_size=tuple(args.target_size))),
        IBVSController(),
        switch_distance=args.switch_distance,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "controller_comparison.csv"
    writer_video = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_video = cv2.VideoWriter(
            str(out_dir / "comparison.mp4"), fourcc, fps, (w, h * 2)
        )

    fields = ["frame", "t", "detected", "controller", "phase",
              "lateral", "forward", "vertical", "yaw", "reason"]
    rows: list[dict] = []

    print(f"Джерело {args.source}: {w}x{h} @ {fps:.1f} fps, ціль={args.shape}")
    print("Керування: q — вихід, пробіл — пауза\n")

    frame_idx = 0
    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            detection = detector.detect_best(frame)
            cmd_i = ibvs.update(detection, frame.shape, dt)
            cmd_p = pbvs.update(detection, frame.shape, dt)
            cmd_h = hybrid.update(detection, frame.shape, dt)

            t = frame_idx * dt
            for ctrl, cmd in (("ibvs", cmd_i), ("pbvs", cmd_p),
                              ("hybrid", cmd_h)):
                rows.append({
                    "frame": frame_idx, "t": round(t, 3),
                    "detected": int(detection is not None),
                    "controller": ctrl, "phase": cmd.phase,
                    "lateral": round(cmd.lateral, 4),
                    "forward": round(cmd.forward, 4),
                    "vertical": round(cmd.vertical, 4),
                    "yaw": round(cmd.yaw, 4),
                    "reason": cmd.reason,
                })

            canvas = draw_comparison(frame, detection, cmd_i, cmd_p, frame_idx, t)

            if writer_video is not None:
                writer_video.write(canvas)
            if not args.no_display:
                cv2.imshow("IBVS (зверху) vs PBVS (знизу)", canvas)

        if not args.no_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
        elif frame_idx % 100 == 0:
            print(f"  оброблено {frame_idx} кадрів")

    cap.release()
    if writer_video is not None:
        writer_video.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    print(f"\nОброблено кадрів: {frame_idx}")
    print(f"CSV: {csv_path}")
    if args.save_video:
        print(f"Відео: {out_dir / 'comparison.mp4'}")
    summarise(rows)
    return 0


def summarise(rows: list[dict]) -> None:
    """Коротке зведення — те, за чим обирається стратегія."""
    if not rows:
        return
    print("\n" + "=" * 62)
    print("Зведення")
    print("=" * 62)

    total_frames = len({r["frame"] for r in rows})
    detected = len({r["frame"] for r in rows if r["detected"]})
    print(f"Кадрів усього: {total_frames}, з детекцією: {detected} "
          f"({detected / max(1, total_frames):.0%})")

    order = {"ibvs": 0, "pbvs": 1, "hybrid": 2}
    present = sorted({r["controller"] for r in rows},
                     key=lambda c: (order.get(c, 99), c))
    for ctrl in present:
        sub = [r for r in rows if r["controller"] == ctrl]
        if not sub:
            continue
        phases: dict[str, int] = {}
        for r in sub:
            phases[r["phase"]] = phases.get(r["phase"], 0) + 1

        mags = [abs(r["lateral"]) + abs(r["forward"])
                + abs(r["vertical"]) + abs(r["yaw"]) for r in sub]
        jitter = float(np.mean([
            abs(sub[i]["lateral"] - sub[i - 1]["lateral"])
            + abs(sub[i]["forward"] - sub[i - 1]["forward"])
            + abs(sub[i]["vertical"] - sub[i - 1]["vertical"])
            for i in range(1, len(sub))
        ])) if len(sub) > 1 else 0.0

        print(f"\n{ctrl.upper()}")
        print(f"  фази: " + ", ".join(f"{k}={v}" for k, v in sorted(phases.items())))
        print(f"  середнє зусилля керування: {np.mean(mags):.3f}")
        print(f"  смикання (менше = плавніше): {jitter:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
