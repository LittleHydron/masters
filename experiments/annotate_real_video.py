"""Прогнати реальне відео крізь детектор і три контролери з накладенням.

Показує на кожному кадрі: детекцію обруча, відновлену позу (дистанцію
й нахил), обидві гілки нормалі — і команди трьох стратегій керування
у вигляді стрілок та чисел.

ВАЖЛИВО про інтерпретацію: відео зняте рукою, а не дроном, тож контур
РОЗІМКНЕНИЙ — команди контролерів нікуди не ведуть, камера рухається
незалежно від них. Тест показує, що команди ОСМИСЛЕНІ (спрямовані до
обруча), але НЕ що дрон долетів би.

Приклад:
    python -m experiments.annotate_real_video \\
        --video datasets/real/Kitchen.mov --calib calibration/iphone14_1920x1080.json
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

from src.control.hybrid import HybridController
from src.control.ibvs import IBVSController
from src.control.pbvs import PBVSController
from src.detection.base import Detection, TargetShape
from src.pose.pnp_solver import PnPSolver
from src.sim.camera_degradation import TelloProfile, degrade

PANEL_W = 460
COLORS = {"ibvs": (90, 200, 255), "pbvs": (140, 255, 140),
          "hybrid": (255, 170, 120)}


def load_calib(path: str | None, w: int, h: int) -> np.ndarray:
    if path:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        K = np.array(d["K"], dtype=np.float64)
        # Калібрування зроблене на іншій роздільності — масштабуємо.
        if d["width"] != w or d["height"] != h:
            K = K.copy()
            K[0, :] *= w / d["width"]
            K[1, :] *= h / d["height"]
        return K
    fx = (w / 2.0) / np.tan(np.deg2rad(82.0) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])


def to_detection(box, conf: float, frame) -> Detection:
    """YOLO-бокс -> Detection з еліпсом, вписаним у бокс.

    Точний контур обруча YOLO не дає, тож еліпс апроксимується боксом.
    Для IBVS цього досить (він працює з формою), для PBVS — джерело
    похибки нахилу, яку варто мати на увазі.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return Detection(
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        confidence=float(conf), shape=TargetShape.CIRCLE,
        ellipse=((float(cx), float(cy)),
                 (float(x2 - x1), float(y2 - y1)), 0.0),
    )


def draw_arrow(img, cx, cy, lat, fwd, vert, color, scale=70):
    """Стрілка напрямку: вбік/вгору — площина, вперед — товщина."""
    dx, dy = lat * scale, -vert * scale
    thick = max(3, int(3 + abs(fwd) * 9))
    if abs(dx) < 2 and abs(dy) < 2:
        cv2.circle(img, (int(cx), int(cy)), 9, color, thick)
    else:
        cv2.arrowedLine(img, (int(cx), int(cy)),
                        (int(cx + dx), int(cy + dy)), color, thick,
                        tipLength=0.3)
    if fwd > 0.05:
        cv2.circle(img, (int(cx), int(cy)), int(6 + fwd * 14), color, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", default="calibration/iphone14_1920x1080.json")
    ap.add_argument("--weights", default="models/yolov8n_mixed.pt")
    ap.add_argument("--diameter", type=float, default=0.89)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--degrade", action="store_true",
                    help="понизити якість до параметрів Tello")
    ap.add_argument("--switch-distance", type=float, default=2.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-frames", type=int, default=0,
                    help="додатково зберігати кожен N-й кадр як JPEG "
                         "(зручніше гортати, ніж відео)")
    ap.add_argument("--out", default="experiments/results_real")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Не відкрити {args.video}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prof = TelloProfile()
    rng = np.random.default_rng(0)
    w, h = (prof.width, prof.height) if args.degrade else (src_w, src_h)

    K = load_calib(args.calib, w, h)
    solver = PnPSolver(K, target_diameter=args.diameter)
    ctrls = {
        "ibvs": IBVSController(),
        "pbvs": PBVSController(PnPSolver(K, target_diameter=args.diameter)),
        "hybrid": HybridController(
            PBVSController(PnPSolver(K, target_diameter=args.diameter)),
            IBVSController(), switch_distance=args.switch_distance),
    }

    stem = Path(args.video).stem + ("_tello" if args.degrade else "")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / f"{stem}_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (w + PANEL_W, h))
    frames_dir = out_dir / f"{stem}_frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    dt = 1.0 / fps
    idx = 0
    detected = 0

    print(f"{Path(args.video).name}: {src_w}x{src_h} -> {w}x{h}"
          f"{' (Tello)' if args.degrade else ''}, {total} кадрів")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if args.max_frames and idx > args.max_frames:
            break
        if args.degrade:
            frame = degrade(frame, prof, rng)
        elif (src_w, src_h) != (w, h):
            frame = cv2.resize(frame, (w, h))

        res = model.predict(frame, conf=args.conf, verbose=False,
                            device=args.device)
        boxes = res[0].boxes
        det = None
        if boxes is not None and len(boxes):
            k = int(np.argmax(boxes.conf.cpu().numpy()))
            det = to_detection(boxes.xyxy[k].tolist(),
                               float(boxes.conf[k]), frame)
            detected += 1

        canvas = np.zeros((h, w + PANEL_W, 3), np.uint8)
        canvas[:, :w] = frame
        pose = solver.solve(det) if det is not None else None

        if det is not None:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.ellipse(canvas, (cx, cy), ((x2 - x1) // 2, (y2 - y1) // 2),
                        0, 0, 360, (0, 230, 0), 1)
            cv2.drawMarker(canvas, (cx, cy), (0, 230, 0),
                           cv2.MARKER_CROSS, 18, 2)
            cv2.putText(canvas, f"{det.confidence:.2f}", (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
            # Центр кадру — ціль для вирівнювання.
            cv2.drawMarker(canvas, (w // 2, h // 2), (110, 110, 110),
                           cv2.MARKER_TILTED_CROSS, 14, 1)

            if pose is not None:
                # Обидві гілки нормалі: суцільна — обрана, пунктир —
                # дзеркальна. Візуалізує двозначність розкладу коніки.
                for br, style in ((pose, True), (pose.alternative, False)):
                    if br is None:
                        continue
                    tip = br.position + br.normal * 0.45
                    if tip[2] > 0.05 and br.position[2] > 0.05:
                        p0 = K @ br.position; p1 = K @ tip
                        a = (int(p0[0] / p0[2]), int(p0[1] / p0[2]))
                        b = (int(p1[0] / p1[2]), int(p1[1] / p1[2]))
                        if style:
                            cv2.arrowedLine(canvas, a, b, (0, 200, 255), 3,
                                            tipLength=0.25)
                        else:
                            for t in np.linspace(0, 1, 9)[::2]:
                                q = (int(a[0] + (b[0] - a[0]) * t),
                                     int(a[1] + (b[1] - a[1]) * t))
                                cv2.circle(canvas, q, 2, (0, 140, 200), -1)

        # Панель праворуч. Шрифти масштабуємо під висоту кадру,
        # інакше на 1080p підписи нечитабельні.
        sc = h / 720.0
        px = w + int(16 * sc)
        F = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"frame {idx}/{total}", (px, int(34 * sc)),
                    F, 0.62 * sc, (190, 190, 190), max(1, int(sc)))
        if pose is not None:
            tilt = np.degrees(np.arccos(np.clip(abs(pose.normal[2]), 0, 1)))
            cv2.putText(canvas, f"d = {pose.distance:.2f} m", (px, int(78 * sc)),
                        F, 0.95 * sc, (0, 210, 255), max(2, int(2 * sc)))
            cv2.putText(canvas, f"tilt = {tilt:.0f} deg", (px, int(112 * sc)),
                        F, 0.68 * sc, (0, 210, 255), max(1, int(sc)))
        else:
            cv2.putText(canvas, "NO TARGET", (px, int(78 * sc)),
                        F, 0.95 * sc, (70, 70, 235), max(2, int(2 * sc)))

        y = int(175 * sc)
        for name, ctrl in ctrls.items():
            cmd = ctrl.update(det, (h, w), dt)
            col = COLORS[name]
            cv2.line(canvas, (px - int(8 * sc), y - int(20 * sc)),
                     (px - int(8 * sc), y + int(52 * sc)), col, max(2, int(3 * sc)))
            cv2.putText(canvas, name.upper(), (px, y),
                        F, 0.78 * sc, col, max(2, int(2 * sc)))
            cv2.putText(canvas, cmd.phase[:24], (px, y + int(28 * sc)),
                        F, 0.55 * sc, (180, 180, 180), max(1, int(sc)))
            cv2.putText(canvas,
                        f"L{cmd.lateral:+.2f}  F{cmd.forward:+.2f}  V{cmd.vertical:+.2f}",
                        (px, y + int(54 * sc)), F, 0.58 * sc, col, max(1, int(sc)))
            draw_arrow(canvas, w + PANEL_W - int(70 * sc), y + int(16 * sc),
                       cmd.lateral, cmd.forward, cmd.vertical, col,
                       scale=int(58 * sc))
            y += int(112 * sc)
            rows.append({
                "frame": idx, "controller": name,
                "detected": int(det is not None),
                "conf": round(det.confidence, 4) if det else 0.0,
                "distance": round(pose.distance, 4) if pose else "",
                "phase": cmd.phase,
                "lateral": round(cmd.lateral, 4),
                "forward": round(cmd.forward, 4),
                "vertical": round(cmd.vertical, 4),
            })

        cv2.putText(canvas, "ROZIMKNENYI KONTUR", (px, h - int(46 * sc)),
                    F, 0.5 * sc, (120, 120, 120), max(1, int(sc)))
        cv2.putText(canvas, "kamera ne sluhaie komand", (px, h - int(22 * sc)),
                    F, 0.44 * sc, (110, 110, 110), max(1, int(sc)))
        writer.write(canvas)
        if args.save_frames and idx % args.save_frames == 0:
            cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"),
                        cv2.resize(canvas, (0, 0), fx=0.5, fy=0.5),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

        if idx % 200 == 0:
            print(f"  {idx}/{total}  детекцій {detected} ({detected/idx:.0%})")

    cap.release()
    writer.release()

    with open(out_dir / f"{stem}_frames.csv", "w", newline="",
              encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\nКадрів: {idx}, з детекцією: {detected} ({detected/max(1,idx):.1%})")
    print(f"Відео: {out_dir / f'{stem}_annotated.mp4'}")
    print(f"CSV:   {out_dir / f'{stem}_frames.csv'}")
    if args.save_frames:
        n = len(list(frames_dir.glob("*.jpg")))
        print(f"Кадри: {frames_dir} ({n} шт.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
