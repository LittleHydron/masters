"""Перевірка масштабу оцінки дистанції на самому дроні.

Питання, на яке відповідає скрипт: чи правильна фокусна відстань,
з якою розв'язувач коніки переводить розмір еліпса в метри?

Метод не потребує шахівниці. Дрон летить прямо на обруч, і ми
порівнюємо ДВІ незалежні величини:

  1. Наскільки змінилася оцінена дистанція (залежить від fx)
  2. Скільки дрон реально пролетів — інтеграл vgx з телеметрії
     (від fx НЕ залежить, це вимір самого дрона)

Їх відношення і є множником помилки fx:

    fx_справжня = fx_використана * (реальний шлях / зміна оцінки)

Якщо оцінка занижує дистанції, дрон "думає", що пролетів менше,
ніж насправді — і відношення буде більшим за одиницю.

БЕЗ ЗЛЬОТУ це не працює: потрібен реальний рух.

Приклад:
    python -m experiments.tello_calibrate_scale --seconds 25
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.control.base import ControlCommand
from src.detection.base import Detection, TargetShape
from src.drone.tello_link import TelloLink
from src.pose.pnp_solver import PnPSolver
from src.viz.live_window import LiveWindow

STUB = (300, 400)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolov8n_mixed2.pt")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--fx", type=float, default=1812.0)
    ap.add_argument("--diameter", type=float, default=0.89)
    ap.add_argument("--max-channel", type=int, default=30)
    ap.add_argument("--stop-distance", type=float, default=2.2,
                    help="припинити зближення на цій ОЦІНЕНІЙ дистанції")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    logging.getLogger("djitellopy").setLevel(logging.WARNING)

    from ultralytics import YOLO
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    model.to(dev)
    model.predict(np.zeros((720, 960, 3), np.uint8), conf=0.25,
                  device=dev, verbose=False)

    link = TelloLink(max_channel=args.max_channel)
    tel = link.connect(stream=True)
    print(tel.summary())

    h, w = 720, 960
    t_wait = time.time()
    while time.time() - t_wait < 10:
        f = link.frame()
        if f is not None and f.shape[:2] != STUB:
            h, w = f.shape[:2]
            break
        time.sleep(0.05)

    fx = args.fx * (w / 960.0)
    K = np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])
    solver = PnPSolver(K, target_diameter=args.diameter)
    win = LiveWindow("scale check", w, h).start() if args.view else None

    samples: list[tuple[float, float, float]] = []
    travelled = 0.0
    last_sig = -1

    print(f"\nЗЛІТ — лечу прямо на обруч до {args.stop_distance} м")
    link.takeoff()
    time.sleep(2.0)
    t0 = last = time.time()
    try:
        while time.time() - t0 < args.seconds:
            now = time.time()
            dt = max(1e-3, now - last)
            last = now
            frame = link.frame()
            if frame is None or frame.shape[:2] == STUB:
                link.watchdog()
                continue
            sig = int(frame[::40, ::40, 0].sum())
            if sig == last_sig:
                continue
            last_sig = sig

            tel = link.telemetry()
            travelled += (tel.speed / 100.0) * dt

            r = model.predict(frame, conf=0.4, device=dev, verbose=False)[0]
            if not len(r.boxes):
                link.hover()
                continue
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[i]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            det = Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(r.boxes.conf.cpu().numpy()[i]),
                shape=TargetShape.CIRCLE,
                ellipse=((float(cx), float(cy)),
                         (float(x2 - x1), float(y2 - y1)), 0.0))
            pose = solver.solve(det)
            if pose is None:
                continue
            d = float(pose.distance)
            samples.append((now - t0, d, travelled))

            ex = (cx - w / 2.0) / (w / 2.0)
            fwd = 0.6 if d > args.stop_distance else 0.0
            link.send(ControlCommand(forward=fwd,
                                     yaw=float(np.clip(ex * 1.2, -0.5, 0.5))))
            if len(samples) % 15 == 0:
                print(f"  {now-t0:5.1f}s  оцінка {d:5.2f} м   "
                      f"пройдено {travelled:5.2f} м   v={tel.speed/100:.2f}")
            if win is not None:
                vis = r.plot()
                cv2.putText(vis, f"d={d:.2f}m  path={travelled:.2f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                win.show(vis)
                if win.quit_requested:
                    break
    finally:
        link.land()
        if win is not None:
            win.close()
        link.close()

    if len(samples) < 20:
        print("замало даних")
        return 1

    a = np.array(samples)
    d0, p0 = a[0, 1], a[0, 2]
    d1, p1 = a[-1, 1], a[-1, 2]
    d_est = d0 - d1
    d_real = p1 - p0

    print("\n" + "=" * 52)
    print(f"оцінка дистанції: {d0:.2f} -> {d1:.2f} м  (зміна {d_est:.2f} м)")
    print(f"реальний шлях за телеметрією:            {d_real:.2f} м")
    if d_est > 0.15:
        ratio = d_real / d_est
        print(f"\nвідношення реальний/оцінений = {ratio:.3f}")
        print(f"поточна fx = {args.fx:.0f}")
        print(f"ВИПРАВЛЕНА fx = {args.fx * ratio:.0f}")
        fov = np.degrees(2 * np.arctan(w / 2 / (args.fx * ratio)))
        print(f"-> поле зору {fov:.1f}° (паспорт Tello ~82°)")
        if abs(ratio - 1.0) < 0.12:
            print("\nМАСШТАБ У НОРМІ (похибка < 12%)")
        else:
            print(f"\nМАСШТАБ ЗМІЩЕНИЙ на {abs(1-ratio)*100:.0f}% — "
                  "перекалібруй камеру Tello шахівницею")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
