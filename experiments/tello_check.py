"""Перша перевірка зв'язку з дроном. БЕЗ ЗЛЬОТУ.

Порядок:
  1. з'єднання по UDP 8889
  2. телеметрія (батарея, висота, температура, орієнтація)
  3. опційно відеопотік: зберігає кадри й проганяє детектор

Дрон стоїть на столі. Пропелери не запускаються.

Приклад:
    python -m experiments.tello_check
    python -m experiments.tello_check --video --seconds 5
    python -m experiments.tello_check --video --weights models/yolov8n_mixed2.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drone.tello_link import TelloLink


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="увімкнути відеопотік")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--weights", default=None, help="перевірити детектор на кадрах")
    ap.add_argument("--out", default="experiments/results/tello_check")
    ap.add_argument("--view", action="store_true", help="вікно з потоком")
    ap.add_argument("--down", action="store_true",
                    help="СПРОБУВАТИ нижню камеру (Robomaster TT НЕ "
                         "підтримує: unknown command: downvision)")
    args = ap.parse_args()

    link = TelloLink()
    print("під'єднуюся до 192.168.10.1 ...")
    try:
        tel = link.connect(stream=args.video)
    except Exception as e:
        print(f"\nНЕ ВДАЛОСЯ: {e}\n")
        print("перевір:")
        print("  * дрон увімкнений, індикатор блимає")
        print("  * ноут у мережі TELLO-XXXXXX (не в домашньому Wi-Fi)")
        print("  * ip addr показує 192.168.10.x на wlan-інтерфейсі")
        return 1

    if args.down:
        print("УВАГА: downvision не підтримується цим дроном, "
              "лишаюсь на передній камері")

    print("\nЗВ'ЯЗОК Є")
    print(tel.summary())

    if not args.video:
        link.close()
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = None
    if args.weights:
        from ultralytics import YOLO
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(args.weights)
        model.to(dev)
        print(f"детектор на {dev}")

    print(f"\nзнімаю {args.seconds:.0f} с ...")
    t0, n, hits = time.time(), 0, 0
    first_shape = None
    while time.time() - t0 < args.seconds:
        frame = link.frame()
        if frame is None:
            continue
        if frame.shape[:2] == (300, 400):
            continue
        n += 1
        if first_shape is None:
            first_shape = frame.shape
        vis = frame
        if model is not None and (args.view or n % 10 == 0):
            r = model.predict(frame, conf=0.25, device=dev, verbose=False)[0]
            vis = r.plot()
            if len(r.boxes) and n % 10 == 0:
                hits += 1
                cv2.imwrite(str(out / f"det_{n:04d}.jpg"), vis)
        if n % 10 == 0:
            cv2.imwrite(str(out / f"frame_{n:04d}.jpg"), frame)
        if args.view:
            cv2.imshow("Tello — stream", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        time.sleep(0.005 if args.view else 0.03)

    dt = time.time() - t0
    print(f"кадрів: {n} за {dt:.1f} с -> {n/dt:.1f} fps")
    print(f"роздільність: {first_shape}")
    if model is not None:
        print(f"детекцій: {hits} із {n//10} перевірених кадрів")
    print(f"збережено в {out}/")
    cv2.destroyAllWindows()
    print(f"\nбатарея після тесту: {link.telemetry().battery}%")
    link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
