"""Демонстрація повного контуру: рендер -> YOLO -> поза -> керування -> рух.

Замкнений контур із СПРАВЖНІМ детектором. На відміну від
`closed_loop_sim.py`, де камера «бачила» проєкцію точок кола, тут на
кожному кроці рендериться зображення, і його обробляє та сама YOLO,
яку ми обирали на реальних відео.

Це найповніша перевірка системи без дрона: кожна ланка справжня, крім
фізики руху (спрощена модель без аеродинаміки) і фону (процедурний).

Вивід — відео, де видно кадр з детекцією, оцінену дистанцію й нахил,
команди трьох стратегій керування та вигляд згори з траєкторією.

Приклад:
    python -m experiments.flight_demo --weights models/yolov8n_mixed2.pt
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
from src.synth.backgrounds import procedural_room
from src.synth.hoop_renderer import HoopSpec, render_hoop

PANEL_W = 460
MAP_H = 260
COLORS = {"ibvs": (90, 200, 255), "pbvs": (140, 255, 140),
          "hybrid": (255, 170, 120)}


def intrinsics(w: int, h: int, fov_deg: float) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])


def world_to_cam(hoop_pos: np.ndarray, hoop_normal: np.ndarray,
                 drone) -> tuple[np.ndarray, np.ndarray]:
    """Позиція й нормаль обруча в системі камери, з урахуванням курсу.

    Дрон обертається під час обльоту, тому нормаль теж треба повернути:
    інакше рендер показував би ракурс, якого дрон не бачить.
    """
    return drone.to_camera(hoop_pos), drone.rotate_to_camera(hoop_normal)


def draw_arrow(img, cx, cy, lat, fwd, vert, color, scale=55):
    dx, dy = lat * scale, -vert * scale
    thick = max(3, int(3 + abs(fwd) * 8))
    if abs(dx) < 2 and abs(dy) < 2:
        cv2.circle(img, (int(cx), int(cy)), 9, color, thick)
    else:
        cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx + dx), int(cy + dy)),
                        color, thick, tipLength=0.3)
    if fwd > 0.05:
        cv2.circle(img, (int(cx), int(cy)), int(7 + fwd * 13), color, 2)


def draw_map(w: int, h: int, trail: list[np.ndarray], cam: np.ndarray,
             hoop_pos: np.ndarray, hoop_normal: np.ndarray,
             radius: float) -> np.ndarray:
    """Вигляд згори: траєкторія й обруч у площині XZ."""
    m = np.full((h, w, 3), 26, np.uint8)
    span_x, z_min, z_max = 4.0, -6.0, 1.5
    def px(p):
        x = int(w / 2 + p[0] / span_x * (w / 2 - 20))
        y = int(h - 15 - (p[2] - z_min) / (z_max - z_min) * (h - 30))
        return x, y

    cv2.putText(m, "vyd zverhu", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (150, 150, 150), 1)
    perp = np.array([-hoop_normal[2], 0.0, hoop_normal[0]])
    n = np.linalg.norm(perp)
    if n > 1e-6:
        perp = perp / n * radius
        cv2.line(m, px(hoop_pos - perp), px(hoop_pos + perp), (60, 90, 240), 4)
    cv2.circle(m, px(hoop_pos), 3, (60, 90, 240), -1)

    for a, b in zip(trail, trail[1:]):
        cv2.line(m, px(a), px(b), (90, 200, 255), 2)
    if trail:
        cv2.circle(m, px(cam), 5, (255, 255, 255), -1)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/yolov8n_mixed2.pt")
    ap.add_argument("--controller", default="hybrid",
                    choices=["ibvs", "pbvs", "hybrid"],
                    help="яка стратегія КЕРУЄ (решта показані для порівняння)")
    ap.add_argument("--diameter", type=float, default=0.89)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=82.0)
    ap.add_argument("--start", nargs=3, type=float, default=[1.1, 0.35, -4.5],
                    metavar=("X", "Y", "Z"))
    ap.add_argument("--tilt", type=float, default=18.0, help="нахил обруча, град")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--latency", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="experiments/results_demo")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    w, h = args.width, args.height
    K = intrinsics(w, h, args.fov)
    radius = args.diameter / 2.0
    rng = np.random.default_rng(args.seed)

    t = np.deg2rad(args.tilt)
    hoop_pos = np.zeros(3)
    hoop_normal = np.array([np.sin(t), 0.0, np.cos(t)])
    spec = HoopSpec(diameter=args.diameter, tube_diameter=0.020,
                    color_bgr=(40, 40, 200))

    background = procedural_room(w, h, rng, distractor_circles=False).astype(np.uint8)

    solver = PnPSolver(K, target_diameter=args.diameter)
    ctrls = {
        "ibvs": IBVSController(),
        "pbvs": PBVSController(PnPSolver(K, target_diameter=args.diameter)),
        "hybrid": HybridController(
            PBVSController(PnPSolver(K, target_diameter=args.diameter)),
            IBVSController(), switch_distance=2.0),
    }
    drone = DroneModel(np.array(args.start, dtype=float),
                       DroneParams(max_speed=args.speed, latency=args.latency),
                       rng)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.weights).stem
    writer = cv2.VideoWriter(str(out_dir / f"flight_{tag}_{args.controller}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 30,
                             (w + PANEL_W, h))
    rows: list[dict] = []
    trail: list[np.ndarray] = []
    dt = 1 / 30.0
    prev_side = float(drone.p @ hoop_normal)
    outcome = "timeout"
    detected = 0

    print(f"Модель: {args.weights}")
    print(f"Керує: {args.controller.upper()}, старт {args.start}, "
          f"нахил обруча {args.tilt}°")

    for step in range(args.max_steps):
        cam = drone.p.copy()
        trail.append(cam.copy())
        rel_pos, rel_n = world_to_cam(hoop_pos, hoop_normal, drone)

        res = render_hoop(background, K, rel_pos, rel_n, spec, rng=rng)
        frame = res.image if res is not None else background.copy()

        pred = model.predict(frame, conf=args.conf, verbose=False,
                             device=args.device)
        boxes = pred[0].boxes
        det = None
        if boxes is not None and len(boxes):
            k = int(np.argmax(boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = boxes.xyxy[k].tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            det = Detection(bbox=(x1, y1, x2, y2),
                            confidence=float(boxes.conf[k]),
                            shape=TargetShape.CIRCLE,
                            ellipse=((cx, cy), (x2 - x1, y2 - y1), 0.0))
            detected += 1

        pose = solver.solve(det) if det is not None else None
        cmds = {n: c.update(det, (h, w), dt) for n, c in ctrls.items()}
        cmd = cmds[args.controller]

        canvas = np.zeros((h, w + PANEL_W, 3), np.uint8)
        canvas[:, :w] = frame
        if det is not None:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cv2.putText(canvas, f"{det.confidence:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
        cv2.drawMarker(canvas, (w // 2, h // 2), (110, 110, 110),
                       cv2.MARKER_TILTED_CROSS, 16, 1)

        px = w + 16
        F = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"krok {step}", (px, 30), F, 0.6, (190, 190, 190), 1)
        true_d = float(np.linalg.norm(rel_pos))
        cv2.putText(canvas, f"istynna d = {true_d:.2f} m", (px, 62),
                    F, 0.6, (150, 150, 150), 1)
        if pose is not None:
            tilt = np.degrees(np.arccos(np.clip(abs(pose.normal[2]), 0, 1)))
            cv2.putText(canvas, f"d = {pose.distance:.2f} m", (px, 100),
                        F, 0.95, (0, 210, 255), 2)
            cv2.putText(canvas, f"tilt = {tilt:.0f} deg", (px, 132),
                        F, 0.6, (0, 210, 255), 1)
        else:
            cv2.putText(canvas, "NO TARGET", (px, 100), F, 0.9, (70, 70, 235), 2)

        y = 185
        for name in ("ibvs", "pbvs", "hybrid"):
            c = cmds[name]
            col = COLORS[name]
            active = (name == args.controller)
            cv2.line(canvas, (px - 9, y - 20), (px - 9, y + 50), col,
                     4 if active else 2)
            label = name.upper() + ("  <= KERUE" if active else "")
            cv2.putText(canvas, label, (px, y), F, 0.7, col,
                        2 if active else 1)
            cv2.putText(canvas, c.phase[:24], (px, y + 26), F, 0.5,
                        (180, 180, 180), 1)
            cv2.putText(canvas,
                        f"L{c.lateral:+.2f} F{c.forward:+.2f} V{c.vertical:+.2f}",
                        (px, y + 50), F, 0.52, col, 1)
            draw_arrow(canvas, w + PANEL_W - 62, y + 14,
                       c.lateral, c.forward, c.vertical, col)
            y += 104

        canvas[h - MAP_H:h, w:] = draw_map(PANEL_W, MAP_H, trail, cam,
                                           hoop_pos, hoop_normal, radius)
        writer.write(canvas)

        rows.append({"step": step, "true_dist": round(true_d, 4),
                     "detected": int(det is not None),
                     "est_dist": round(pose.distance, 4) if pose else "",
                     "phase": cmd.phase,
                     "lateral": round(cmd.lateral, 4),
                     "forward": round(cmd.forward, 4),
                     "vertical": round(cmd.vertical, 4)})

        drone.step(cmd, dt)
        side = float(drone.p @ hoop_normal)
        if prev_side < 0 <= side:
            a = prev_side / (prev_side - side) if prev_side != side else 0.0
            hit = cam + (drone.p - cam) * a
            radial = float(np.linalg.norm(hit - hoop_normal * float(hit @ hoop_normal)))
            outcome = "PROLIT" if radial < radius else "PROMAH"
            print(f"  крок {step}: {outcome}, промах {radial - radius:+.2f} м")
            for _ in range(15):
                writer.write(canvas)
            break
        prev_side = side
        if np.linalg.norm(drone.p) > 12.0:
            outcome = "rozlit"
            break

    writer.release()
    with open(out_dir / f"flight_{tag}_{args.controller}.csv", "w",
              newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    print(f"Результат: {outcome}, кроків {len(rows)}, "
          f"детекцій {detected}/{len(rows)} ({detected/max(1,len(rows)):.0%})")
    print(f"Відео: {out_dir / f'flight_{tag}_{args.controller}.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
