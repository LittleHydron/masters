"""Латентність на СПРАВЖНЬОМУ дроні: виміряти і показати її вплив.

Два досліди в одному скрипті. Обидва потребують польоту — на столі
затримки не видно, бо немає руху.

ДОСЛІД 1 (--mode measure): скільки саме запізнюється ВІДЕО.
    Дрон висить перед обручем, отримує різкий імпульс по курсу.

    Міряємо НЕ "коли дрон зрушив" — це інерція, а не затримка
    (перший захід 29.09 дав 498 мс, і один замір показав камеру
    РАНІШЕ за телеметрію, що фізично неможливо: поріг спрацьовував
    на розгоні, а не на запізненні).

    Міряємо ЗСУВ У ЧАСІ між двома сигналами одного й того самого
    руху: курс із телеметрії та положення цілі в кадрі. Обидва
    пишемо з мітками часу, потім шукаємо затримку, за якої їхні
    похідні найкраще збігаються (крос-кореляція). Це і є затримка
    відеотракту: H.264, Wi-Fi, декодування, інференс.

ДОСЛІД 2 (--mode effect): що змінює компенсація.
    Дрон стежить за обручем по курсу (як `tello_fly --mode track`),
    але ціль РУХАЄТЬСЯ — її носить користувач. Половину часу
    компенсація увімкнена, половину вимкнена, з чергуванням, щоб
    умови були однакові. Міряємо:
      * смикання — середня |зміна команди| між кадрами: пряма міра
        перерегулювання від запізнілої реакції;
      * |помилка| — наскільки ціль відхилена від центру кадру.

Чергування, а не два окремі польоти: батарея сідає, а з нею
змінюється поведінка дрона — послідовні заміри були б непорівнянні.

Приклад:
    python -m experiments.latency_on_drone --mode measure --pulses 6
    python -m experiments.latency_on_drone --mode effect --seconds 60 --view
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.control.base import PID, ControlCommand
from src.control.latency import make_predictor
from src.drone.tello_link import TelloLink
from src.viz.live_window import LiveWindow

STUB = (300, 400)


def load_model(weights: str):
    from ultralytics import YOLO
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = YOLO(weights)
    m.to(dev)
    m.predict(np.zeros((720, 960, 3), np.uint8), conf=0.25,
              device=dev, verbose=False)
    return m, dev


def target_cx(model, dev, frame, conf=0.4):
    """Горизонтальний центр найвпевненішої детекції, або None."""
    r = model.predict(frame, conf=conf, device=dev, verbose=False)[0]
    if not len(r.boxes):
        return None, r
    i = int(np.argmax(r.boxes.conf.cpu().numpy()))
    x1, _, x2, _ = r.boxes.xyxy.cpu().numpy()[i]
    return float((x1 + x2) / 2.0), r


def run_measure(link, model, dev, w, h, args) -> int:
    """Затримка відео = часовий зсув між курсом і картинкою."""
    print(f"\nЗЛІТ — {args.pulses} імпульсів по курсу")
    link.takeoff()
    time.sleep(3.0)

    series: list[tuple[float, float, float]] = []
    try:
        for k in range(args.pulses):
            link.hover()
            time.sleep(args.settle)
            direction = 1.0 if k % 2 == 0 else -1.0
            t_cmd = time.time()
            link.send(ControlCommand(yaw=direction * args.pulse_yaw))
            while time.time() - t_cmd < args.pulse_len + 1.5:
                if time.time() - t_cmd > args.pulse_len:
                    link.hover()
                f = link.frame()
                if f is None or f.shape[:2] == STUB:
                    continue
                cx, _ = target_cx(model, dev, f)
                if cx is None:
                    continue
                series.append((time.time(), float(link.telemetry().yaw), cx))
            link.hover()
            print(f"  імпульс {k+1}: {len(series)} точок")
    finally:
        link.land()

    if len(series) < 60:
        print("\nзамало точок")
        return 1

    a = np.array(series)
    t, yaw, cx = a[:, 0] - a[0, 0], a[:, 1], a[:, 2]

    step = 0.02
    grid = np.arange(0, t[-1], step)
    yaw_u = np.interp(grid, t, np.unwrap(np.deg2rad(yaw)))
    cx_u = np.interp(grid, t, cx)

    dy = np.diff(yaw_u)
    dc = np.diff(cx_u)
    dy = (dy - dy.mean()) / (dy.std() + 1e-9)
    dc = (dc - dc.mean()) / (dc.std() + 1e-9)

    max_lag = int(args.max_lag / step)
    lags = np.arange(0, max_lag)
    corr = [float(np.dot(dy[:len(dy) - L], dc[L:])) / max(1, len(dy) - L)
            for L in lags]
    best = int(np.argmin(corr))
    lag_s = best * step

    print("\n" + "=" * 56)
    print(f"точок: {len(series)}, вікно {t[-1]:.1f} с")
    print(f"ЗАТРИМКА ВІДЕО: {lag_s*1000:.0f} мс "
          f"(кореляція {corr[best]:+.2f})")
    if corr[best] > -0.15:
        print("УВАГА: кореляція слабка — рух був замалий або "
              "детекція нестабільна. Повтори з більшим --pulse-yaw.")
    print("=" * 56)
    print(f"\n-> використовувати: --latency {lag_s:.2f}")

    out = Path(args.out) / "latency_measured.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": "cross-correlation yaw-rate vs pixel-rate",
        "points": len(series),
        "video_latency_ms": round(lag_s * 1000, 1),
        "correlation": round(corr[best], 3),
        "curve": [[round(L * step, 3), round(c, 3)]
                  for L, c in zip(lags, corr)],
    }, indent=2))
    print(f"збережено: {out}")
    return 0


def run_effect(link, model, dev, w, h, args, win) -> int:
    """Стеження за рухомою ціллю з чергуванням компенсації."""
    pid = PID(args.kp_yaw, 0.0, 0.15)
    pred = make_predictor("kalman", 2)
    rows: list[dict] = []
    stop = {"flag": False}

    def on_sigint(sig, frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, on_sigint)

    print(f"\nЗЛІТ — стеження, компенсація чергується кожні "
          f"{args.block:.0f} с")
    print("РУХАЙ ОБРУЧ вліво-вправо: без руху затримки не видно")
    link.takeoff()
    time.sleep(2.5)

    t0 = last = time.time()
    prev_cmd = 0.0
    last_sig = -1
    pred_t = 0.0
    try:
        while time.time() - t0 < args.seconds and not stop["flag"]:
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

            cx, res = target_cx(model, dev, frame, conf=0.5)
            if cx is None:
                link.hover()
                continue

            block = int((now - t0) // args.block)
            comp_on = (block % 2 == 1)

            pred_t += dt
            pred.update(np.array([cx, h / 2.0]), pred_t)
            cx_used = cx
            shift = 0.0
            if comp_on:
                p = pred.predict(args.latency)
                if p is not None:
                    cx_used = float(p[0])
                    shift = cx_used - cx

            err = (cx_used - w / 2.0) / (w / 2.0)
            err_true = (cx - w / 2.0) / (w / 2.0)
            yaw = float(np.clip(pid(err, dt), -1.0, 1.0))
            rc = link.send(ControlCommand(yaw=yaw))

            rows.append({
                "t": round(now - t0, 3), "comp": int(comp_on),
                "err_true": round(err_true, 4),
                "shift_px": round(shift, 1),
                "yaw": rc[3],
                "d_yaw": abs(rc[3] - prev_cmd),
            })
            prev_cmd = rc[3]

            if win is not None:
                vis = res.plot()
                lbl = "COMP ON" if comp_on else "comp off"
                col = (0, 220, 60) if comp_on else (160, 160, 160)
                cv2.putText(vis, f"{lbl}  err={err_true:+.2f}  "
                            f"shift={shift:+.0f}px", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                win.show(vis)
                if win.quit_requested:
                    break
    finally:
        link.land()

    if len(rows) < 40:
        print("замало даних")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "latency_effect.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print("\n" + "=" * 56)
    print(f"{'':12s} {'кадрів':>7s} {'смикання':>10s} {'|помилка|':>11s}")
    stats = {}
    for on in (0, 1):
        sub = [r for r in rows if r["comp"] == on]
        if not sub:
            continue
        jit = float(np.mean([r["d_yaw"] for r in sub]))
        err = float(np.median([abs(r["err_true"]) for r in sub]))
        stats[on] = (jit, err, len(sub))
        name = "з компенс." if on else "без компенс."
        print(f"{name:12s} {len(sub):>7d} {jit:>10.2f} {err:>11.3f}")
    print("=" * 56)
    if 0 in stats and 1 in stats:
        dj = (stats[1][0] - stats[0][0]) / max(1e-6, stats[0][0]) * 100
        de = (stats[1][1] - stats[0][1]) / max(1e-6, stats[0][1]) * 100
        print(f"смикання: {dj:+.0f}%   |помилка|: {de:+.0f}%")
        print("(від'ємне = компенсація допомогла)")
    print(f"\nлог: {out / 'latency_effect.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "effect"], default="measure")
    ap.add_argument("--weights", default="models/yolov8n_mixed2.pt")
    ap.add_argument("--max-channel", type=int, default=45)
    ap.add_argument("--out", default="experiments/results/latency")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--pulses", type=int, default=6)
    ap.add_argument("--pulse-yaw", type=float, default=1.0)
    ap.add_argument("--pulse-len", type=float, default=0.45)
    ap.add_argument("--settle", type=float, default=2.5)
    ap.add_argument("--max-lag", type=float, default=0.8,
                    help="у якому діапазоні шукати затримку, с")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--block", type=float, default=6.0,
                    help="довжина блоку чергування, с")
    ap.add_argument("--latency", type=float, default=0.20,
                    help="на скільки екстраполювати (з досліду 1)")
    ap.add_argument("--kp-yaw", type=float, default=1.2)
    args = ap.parse_args()

    logging.getLogger("djitellopy").setLevel(logging.WARNING)
    model, dev = load_model(args.weights)
    print(f"детектор на {dev}")

    link = TelloLink(max_channel=args.max_channel)
    tel = link.connect(stream=True)
    print(tel.summary())
    if tel.temp_c > 85:
        print("ВІДМОВА: перегрів")
        link.close()
        return 1

    h, w = 720, 960
    t_wait = time.time()
    while time.time() - t_wait < 10:
        f = link.frame()
        if f is not None and f.shape[:2] != STUB:
            h, w = f.shape[:2]
            break
        time.sleep(0.05)
    print(f"кадр: {w}x{h}")

    win = LiveWindow("latency", w, h).start() if args.view else None
    try:
        if args.mode == "measure":
            return run_measure(link, model, dev, w, h, args)
        return run_effect(link, model, dev, w, h, args, win)
    finally:
        if win is not None:
            win.close()
        link.close()


if __name__ == "__main__":
    raise SystemExit(main())
