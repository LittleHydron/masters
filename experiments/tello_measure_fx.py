"""Вимір фокусної відстані камери Tello за обручем відомого діаметра.

Навіщо: PBVS переводить розмір обруча в кадрі на метри, і для цього
потрібна `fx`. Без власного виміру вона береться з паспортного FOV
82°, що дає систематичну похибку дистанції — а разом із нею пливе
поріг перемикання гібрида на 2 м.

Метод — пряма геометрія тонкої лінзи. Для об'єкта відомого розміру:

    fx = (розмір_у_пікселях * дистанція_в_метрах) / розмір_у_метрах

Дрон НЕ злітає: стоїть на столі чи підлозі й дивиться на обруч.
Це головна перевага над `tello_calibrate_scale.py`, який виводить
масштаб з інтеграла швидкості в польоті — там похибка телеметрії
накладається на похибку оцінки.

Порядок роботи:
  1. постав обруч вертикально, дрон — навпроти центра обруча;
  2. виміряй рулеткою відстань від ОБ'ЄКТИВА до площини обруча;
  3. запусти скрипт із цією відстанню, дочекайся N кадрів;
  4. повтори з 2-3 різних відстаней — скрипт зведе їх разом.

Що показує узгодженість: якщо `fx` з різних відстаней збігається в
межах кількох відсотків, вимір надійний. Якщо розходиться — обруч
стоїть не фронтально або відстань виміряна не до об'єктива.

Приклад:
    python -m experiments.tello_measure_fx --distance 2.0
    python -m experiments.tello_measure_fx --distance 3.0
    python -m experiments.tello_measure_fx --summary
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drone.tello_link import TelloLink
from src.viz.live_window import LiveWindow

STUB = (300, 400)
OUT = Path("experiments/results/tello_fx.json")


def summarize() -> int:
    """Звести всі накопичені заміри в одну оцінку."""
    if not OUT.exists():
        print(f"немає {OUT} — спершу зроби хоча б один вимір")
        return 1
    data = json.loads(OUT.read_text(encoding="utf-8"))
    if not data:
        print("файл порожній")
        return 1

    print(f"\n{'дистанція':>10} {'кадрів':>7} {'рамка, px':>11} {'fx':>8}")
    print("-" * 40)
    for r in data:
        print(f"{r['distance_m']:10.2f} {r['frames']:7d} "
              f"{r['box_px_median']:11.1f} {r['fx']:8.0f}")

    fxs = [r["fx"] for r in data]
    fx = st.median(fxs)
    print("-" * 40)
    print(f"{'МЕДІАНА':>10} {'':7} {'':11} {fx:8.0f}")

    if len(fxs) > 1:
        spread = (max(fxs) - min(fxs)) / fx * 100
        print(f"\nрозкид між замірами: {spread:.1f}%")
        if spread > 8:
            print("  ЗАВЕЛИКИЙ — обруч, імовірно, стоїть не фронтально,")
            print("  або відстань виміряна не до об'єктива. Переміряй.")
        else:
            print("  у нормі")
    else:
        print("\nзамір ОДИН — зроби ще з іншої відстані для перевірки")

    w = data[0]["frame_w"]
    fov = np.degrees(2 * np.arctan(w / 2.0 / fx))
    print(f"\nполе зору по горизонталі: {fov:.1f}° (паспорт Tello ~82°)")
    print(f"\nВИКОРИСТАННЯ:")
    print(f"  python -m experiments.tello_fly --mode approach --fx {fx:.0f}")
    return 0


def draw_aim(frame, box, conf, w: int, h: int, diameter: float,
             distance: float, collected: int, target: int,
             collecting: bool) -> np.ndarray:
    """Екран вирівнювання: усе, що потрібно, щоб виставити дрон.

    Показує не лише рамку, а й ДВІ величини, які псують вимір:
    нахил обруча навколо горизонтальної осі (стискає висоту, за якою
    рахується fx) і зміщення цілі від центра кадру. Обидві — з
    оцінкою похибки у відсотках, щоб було видно, коли досить.
    """
    vis = frame.copy()
    cx_f, cy_f = w // 2, h // 2

    cv2.drawMarker(vis, (cx_f, cy_f), (0, 0, 0), cv2.MARKER_CROSS, 40, 4)
    cv2.drawMarker(vis, (cx_f, cy_f), (255, 255, 255), cv2.MARKER_CROSS, 38, 1)
    cv2.circle(vis, (cx_f, cy_f), 110, (70, 70, 70), 1)

    lines: list[tuple[str, tuple[int, int, int]]] = []
    good_tilt = good_off = False

    if box is not None:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ratio = bw / max(1.0, bh)
        tilt = (np.degrees(np.arccos(min(1.0, 1.0 / ratio)))
                if ratio > 1.0 else 0.0)
        off = float(np.hypot(cx - cx_f, cy - cy_f))
        fx_now = bh * distance / diameter
        off_ang = np.degrees(np.arctan(off / max(1.0, fx_now)))

        good_tilt = tilt <= 12.0
        good_off = off_ang <= 15.0
        col = ((0, 220, 60) if (good_tilt and good_off and conf >= 0.7)
               else (0, 200, 255))

        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.circle(vis, (int(cx), int(cy)), 5, col, -1)
        cv2.line(vis, (cx_f, cy_f), (int(cx), int(cy)), col, 1)

        tilt_err = (np.cos(np.deg2rad(tilt)) - 1.0) * 100
        off_err = (np.cos(np.deg2rad(off_ang)) - 1.0) * 100
        lines = [
            (f"рамка  {bh:.0f} px  ->  fx = {fx_now:.0f}", (235, 235, 235)),
            (f"нахил  {tilt:4.0f}°   ({tilt_err:+.1f}%)"
             + ("   ок" if good_tilt else "   ПІДНІМИ/ОПУСТИ ДРОН"),
             (0, 220, 60) if good_tilt else (60, 60, 255)),
            (f"зсув   {off_ang:4.0f}°   ({off_err:+.1f}%)"
             + ("   ок" if good_off else "   ДОВЕРНИ НА ОБРУЧ"),
             (0, 220, 60) if good_off else (60, 60, 255)),
            (f"conf   {conf:.2f}", (235, 235, 235)),
        ]
    else:
        lines = [("обруча не видно", (60, 60, 255))]

    ph = 34 + 26 * len(lines)
    top = h - 38 - ph
    cv2.rectangle(vis, (0, top), (440, h - 38), (0, 0, 0), -1)
    cv2.putText(vis, f"d = {distance:.2f} м   діаметр {diameter:.2f} м",
                (10, top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(vis, txt, (10, top + 50 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1)

    cv2.rectangle(vis, (0, h - 38), (w, h), (0, 0, 0), -1)
    if collecting:
        frac = collected / max(1, target)
        cv2.rectangle(vis, (10, h - 26), (10 + int(300 * frac), h - 12),
                      (0, 220, 60), -1)
        cv2.rectangle(vis, (10, h - 26), (310, h - 12), (90, 90, 90), 1)
        msg = f"ЗБИРАЮ  {collected}/{target}"
        col = (0, 220, 60)
    elif box is not None and good_tilt and good_off:
        msg = "ВИРІВНЯНО — ПРОБІЛ, щоб зібрати кадри"
        col = (0, 220, 60)
    else:
        msg = "вирівняй дрон, потім ПРОБІЛ   (Q — вихід)"
        col = (0, 200, 255)
    cv2.putText(vis, msg, (330 if collecting else 10, h - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float,
                    help="виміряна рулеткою відстань від об'єктива до "
                         "площини обруча, метри")
    ap.add_argument("--diameter", type=float, default=0.89,
                    help="діаметр обруча, метри")
    ap.add_argument("--frames", type=int, default=40,
                    help="скільки кадрів із детекцією зібрати")
    ap.add_argument("--weights", default="models/yolov8n_mixed2.pt")
    ap.add_argument("--min-conf", type=float, default=0.70)
    ap.add_argument("--view", action="store_true", default=True)
    ap.add_argument("--no-view", dest="view", action="store_false")
    ap.add_argument("--summary", action="store_true",
                    help="звести накопичені заміри й вийти")
    args = ap.parse_args()

    if args.summary:
        return summarize()
    if not args.distance:
        print("вкажи --distance (метри) або --summary")
        return 1

    logging.getLogger("djitellopy").setLevel(logging.WARNING)

    from ultralytics import YOLO
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    model.to(dev)
    model.predict(np.zeros((720, 960, 3), np.uint8), conf=0.25,
                  device=dev, verbose=False)
    print(f"детектор: {args.weights} на {dev}")

    link = TelloLink()
    print("під'єднуюся ...")
    tel = link.connect(stream=True)
    print(f"батарея {tel.battery}% | темп {tel.temp_c:.0f}°C")

    w = h = 0
    t_wait = time.time()
    while time.time() - t_wait < 12.0:
        f = link.frame()
        if f is not None and f.shape[:2] != STUB:
            h, w = f.shape[:2]
            break
        time.sleep(0.3)
    if not w:
        print("відео не піднялося")
        link.close()
        return 1
    print(f"кадр: {w}x{h}")

    win = LiveWindow("fx — вирівнювання", w, h).start() if args.view else None
    print(f"\nДРОН НЕ ЗЛІТАЄ. Постав його навпроти обруча на "
          f"{args.distance} м.")
    if win is not None:
        print("У вікні: перехрестя центра кадру, рамка детекції,")
        print("нахил і зсув із похибкою. Коли все зелене — ПРОБІЛ.")
        print("Q або ESC — вихід.\n")
    else:
        print(f"Збираю {args.frames} кадрів без вікна.\n")

    heights: list[float] = []
    widths: list[float] = []
    offsets: list[float] = []
    last_sig = -1
    collecting = win is None
    t0 = time.time()
    try:
        while len(heights) < args.frames and time.time() - t0 < 300:
            if win is not None:
                if win.quit_requested:
                    break
                if win.take_key() == ord(" ") and not collecting:
                    collecting = True
                    print("збираю кадри ...")

            frame = link.frame()
            if frame is None or frame.shape[:2] == STUB:
                time.sleep(0.02)
                continue
            sig = int(frame[::40, ::40, 0].sum())
            if sig == last_sig:
                time.sleep(0.01)
                continue
            last_sig = sig

            r = model.predict(frame, conf=0.25, device=dev, verbose=False)[0]
            box = None
            conf = 0.0
            if len(r.boxes):
                k = int(np.argmax(r.boxes.conf.cpu().numpy()))
                conf = float(r.boxes.conf.cpu().numpy()[k])
                box = tuple(float(v) for v in r.boxes.xyxy.cpu().numpy()[k])

            if collecting and box is not None and conf >= args.min_conf:
                x1, y1, x2, y2 = box
                heights.append(y2 - y1)
                widths.append(x2 - x1)
                offsets.append(float(np.hypot((x1 + x2) / 2.0 - w / 2.0,
                                              (y1 + y2) / 2.0 - h / 2.0)))

            if win is not None:
                win.show(draw_aim(frame, box, conf, w, h, args.diameter,
                                  args.distance, len(heights), args.frames,
                                  collecting))
    finally:
        link.close()
        if win is not None:
            win.close()

    if len(heights) < 10:
        print(f"зібрано лише {len(heights)} кадрів — замало. "
              "Перевір, чи обруч у кадрі й чи достатньо світла.")
        return 1

    bh = st.median(heights)
    bw = st.median(widths)
    fx = bh * args.distance / args.diameter

    print("\n" + "=" * 52)
    print(f"кадрів із детекцією : {len(heights)}")
    print(f"висота рамки, медіана: {bh:.1f} px  (розкид "
          f"{min(heights):.0f}…{max(heights):.0f})")
    print(f"ширина рамки, медіана: {bw:.1f} px")
    ratio = bw / bh
    off = st.median(offsets)

    tilt = (np.degrees(np.arccos(min(1.0, 1.0 / ratio)))
            if ratio > 1.0 else 0.0)
    off_ang = np.degrees(np.arctan(off / fx))
    off_err = (np.cos(np.deg2rad(off_ang)) - 1.0) * 100

    print(f"співвідношення сторін: {ratio:.3f}", end="")
    if tilt > 1.0:
        print(f"  -> нахил ~{tilt:.0f}° (похибка fx "
              f"{(np.cos(np.deg2rad(tilt)) - 1) * 100:+.1f}%)")
    else:
        print("  -> обруч фронтально")
    print(f"зміщення від центра : {off:.0f} px ({off_ang:.0f}°, "
          f"похибка {off_err:+.1f}%)")

    warn = []
    if tilt > 12.0:
        warn.append(f"обруч нахилений на ~{tilt:.0f}° навколо "
                    "горизонтальної осі — підніми або опусти дрон, "
                    "щоб об'єктив дивився в площину обруча прямо")
    if off_ang > 15.0:
        warn.append(f"обруч зміщений на {off_ang:.0f}° від осі — "
                    "доверни дрон, щоб ціль була ближче до центра кадру")
    if warn:
        print("\n  ПЕРЕСТАВ І ПЕРЕМІРЯЙ:")
        for ww in warn:
            print(f"    * {ww}")
    print(f"\nfx = {bh:.1f} px * {args.distance} м / {args.diameter} м"
          f" = {fx:.0f} px")
    fov = np.degrees(2 * np.arctan(w / 2.0 / fx))
    print(f"поле зору по горизонталі: {fov:.1f}° (паспорт Tello ~82°)")
    print("=" * 52)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    data = [r for r in data if abs(r["distance_m"] - args.distance) > 0.01]
    data.append({
        "distance_m": args.distance,
        "diameter_m": args.diameter,
        "frames": len(heights),
        "box_px_median": round(bh, 1),
        "box_w_px_median": round(bw, 1),
        "aspect_ratio": round(ratio, 3),
        "tilt_deg": round(tilt, 1),
        "offset_px": round(off, 1),
        "fx": round(fx, 1),
        "frame_w": w, "frame_h": h,
    })
    data.sort(key=lambda r: r["distance_m"])
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nзбережено: {OUT}")
    print("Зроби ще вимір з іншої відстані, потім:")
    print("  python -m experiments.tello_measure_fx --summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
