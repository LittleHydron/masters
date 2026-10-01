"""Замкнений контур на справжньому дроні: кадр -> YOLO -> поза -> керування.

Режими навмисно розділені за ризиком:

  --mode hover     зліт, висить, сідає. Керування не подається.
                   Перевіряє зліт/посадку й стабільність висіння.

  --mode track     висить і крутиться за обручем ТІЛЬКИ по курсу.
                   Поступального руху немає. Перевіряє, чи замкнувся
                   контур: детекція -> команда -> реакція дрона.

  --mode approach  повний контур із обраним контролером. Дрон
                   рухається. Тільки на відкритому просторі.

СТРАТЕГІЯ ЗА ЗАМОВЧУВАННЯМ — ГІБРИД:

  далі за `--switch-distance` (2 м)  -> PBVS, керування за метричною
                                        позою з розкладу коніки;
  ближче                             -> IBVS, керування в пікселях.

Перемикання незворотне: щойно перейшли на IBVS, назад не повертаємось.
Інакше на межі дрон смикався б між двома законами керування.

Коли обруч перестає вміщатися в кадр, поза вже не відновлюється —
тоді вмикається сліпий проліт на `--blind-seconds` уперед.

Безпека:
  * `--max-channel` обмежує агресивність (типово 40)
  * watchdog обнуляє канали, якщо кадр не прийшов
  * `--seconds` жорстко садить після N секунд
  * Ctrl+C садить, а не кидає дрон
  * ліміт по температурі: >85°C — відмова злітати

ВАЖЛИВО про калібрування: PBVS рахує дистанцію з розкладу коніки, а
отже потребує фокусної відстані камери. Виміряти для Tello:
`python -m experiments.tello_calibrate_scale`. Без власного виміру
дистанція буде систематично хибною, і поріг 2 м попливе.

Приклад:
    python -m experiments.tello_fly --mode hover --seconds 6
    python -m experiments.tello_fly --mode approach --view --record
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.control.base import ControlCommand, PID
from src.control.hybrid import HybridController
from src.control.ibvs import IBVSController
from src.control.latency import make_predictor
from src.control.latency_estimator import LatencyEstimator
from src.control.pbvs import PBVSController
from src.detection.base import Detection, TargetShape
from src.drone.tello_link import TelloLink
from src.pose.pnp_solver import PnPSolver
from src.viz.live_window import LiveWindow

TEMP_LIMIT = 85.0
STUB_SHAPE = (300, 400)
WIN = "Tello — detection"


def build_detection(result, frame_shape) -> Detection | None:
    """Найвпевненіша рамка YOLO -> наш Detection."""
    if not len(result.boxes):
        return None
    i = int(np.argmax(result.boxes.conf.cpu().numpy()))
    x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[i]
    conf = float(result.boxes.conf.cpu().numpy()[i])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return Detection(
        bbox=(float(x1), float(y1), float(x2), float(y2)),
        confidence=conf,
        shape=TargetShape.CIRCLE,
        ellipse=((float(cx), float(cy)),
                 (float(x2 - x1), float(y2 - y1)), 0.0),
    )


def draw_overlay(frame, det, pose, cmd, rc, tel, fps: float,
                 mode: str) -> np.ndarray:
    """Кадр + рамка детекції + телеметрія. Те саме, що бачить контролер."""
    img = frame.copy()
    h, w = img.shape[:2]

    cv2.drawMarker(img, (w // 2, h // 2), (0, 0, 0), cv2.MARKER_CROSS, 28, 3)
    cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255),
                   cv2.MARKER_CROSS, 26, 1)

    if det is not None:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        c = (0, 220, 60) if det.confidence > 0.7 else (
            (0, 200, 255) if det.confidence > 0.45 else (60, 60, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(img, (cx, cy), 4, c, -1)
        cv2.line(img, (w // 2, h // 2), (cx, cy), c, 1)
        cv2.putText(img, f"{det.confidence:.2f}", (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    cv2.rectangle(img, (0, 0), (w, 54), (0, 0, 0), -1)
    left = f"{mode}  {fps:4.1f}fps"
    if pose is not None:
        left += f"  d={pose.distance:.2f}m"
    cv2.putText(img, left, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    if cmd is not None:
        cv2.putText(img, f"{cmd.phase[:34]}  rc={rc}", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1)

    bx = w - 232
    bat_col = ((120, 255, 120) if tel.battery > 40 else
               (0, 215, 255) if tel.battery > 20 else (60, 60, 255))
    cv2.putText(img, f"BAT {tel.battery}%", (bx, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, bat_col, 2)
    cv2.rectangle(img, (bx, 27), (bx + 92, 34), (70, 70, 70), 1)
    cv2.rectangle(img, (bx + 1, 28),
                  (bx + 1 + int(90 * tel.battery / 100.0), 33), bat_col, -1)

    t_col = ((120, 255, 120) if tel.temp_c < 70 else
             (0, 215, 255) if tel.temp_c < 82 else (60, 60, 255))
    cv2.putText(img, f"{tel.temp_c:.0f}C", (bx + 104, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, t_col, 2)
    if tel.temp_c >= 82:
        cv2.putText(img, "HOT", (bx + 104, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 255), 1)

    cv2.putText(img, f"H {tel.height_cm}cm  tof {tel.tof_cm}  "
                     f"v {tel.speed:.0f}cm/s  t{tel.flight_time}s",
                (bx - 4, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (190, 190, 190), 1)

    hx, hy, hr = w - 44, h - 46, 28
    cv2.circle(img, (hx, hy), hr, (90, 90, 90), 1)
    a = np.deg2rad(tel.roll)
    dx, dy = int(hr * np.cos(a)), int(hr * np.sin(a))
    off = int(tel.pitch * 0.5)
    cv2.line(img, (hx - dx, hy - dy + off), (hx + dx, hy + dy + off),
             (120, 220, 255), 2)
    cv2.putText(img, f"r{tel.roll:+d} p{tel.pitch:+d}", (hx - 42, hy + hr + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    if rc is not None:
        base_y = h - 14
        for i, (lbl, v) in enumerate(zip("LFVY", rc)):
            x = 14 + i * 74
            cv2.putText(img, lbl, (x, base_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (150, 150, 150), 1)
            bar = int(abs(v) / 100.0 * 46)
            col = (60, 60, 255) if abs(v) >= 99 else (200, 200, 90)
            cv2.rectangle(img, (x + 16, base_y - 9),
                          (x + 16 + bar, base_y - 2), col, -1)
            cv2.putText(img, f"{v:+d}", (x + 16, base_y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    return img


def track_yaw(det, w: int, pid: PID, dt: float) -> ControlCommand:
    """Режим track: тримати ціль у центрі кадру ТІЛЬКИ курсом.

    Окремий ПІД, а не `cmd.yaw` від контролера: у PBVS курс служить
    для обльоту по дузі й у фазі ORBIT майже завжди однакового знаку.
    Для перевірки замкненості контуру потрібне саме центрування.
    """
    cx = det.center[0]
    ex = (cx - w / 2.0) / (w / 2.0)
    yaw = float(np.clip(pid(ex, dt), -1.0, 1.0))
    return ControlCommand(yaw=yaw, phase=f"TRACK(x={ex:+.2f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hover", "track", "approach"],
                    default="hover")
    ap.add_argument("--weights", default="models/yolov8n_mixed2.pt")
    ap.add_argument("--device", default="cuda",
                    help="cuda або cpu (на CPU контур просідає до ~5 Гц)")
    ap.add_argument("--seconds", type=float, default=25.0,
                    help="жорсткий ліміт польоту")
    ap.add_argument("--max-channel", type=int, default=40,
                    help="стеля rc-каналу: обмежує агресивність")
    ap.add_argument("--min-conf", type=float, default=0.55,
                    help="нижча впевненість -> команда не подається")

    ap.add_argument("--strategy", choices=["hybrid", "pbvs", "ibvs"],
                    default="hybrid")
    ap.add_argument("--switch-distance", type=float, default=2.0,
                    help="гібрид: далі цього PBVS, ближче IBVS")
    ap.add_argument("--diameter", type=float, default=0.89,
                    help="діаметр обруча, м — масштаб для оцінки пози")
    ap.add_argument("--fx", type=float, default=None,
                    help="фокусна відстань у пікселях. Без неї береться "
                         "з горизонтального FOV 82° (паспорт Tello), що "
                         "дає систематичну похибку дистанції. Виміряти: "
                         "python -m experiments.tello_calibrate_scale")
    ap.add_argument("--min-lateral-channel", type=float, default=8.0,
                    help="мінімальний РОБОЧИЙ бічний канал. У Tello "
                         "команда нижче ~7 не зрушує дрона: політ "
                         "01.10 показав lat=1 на останніх 2 м, тобто "
                         "нуль бічного руху — зміщення ex=+0.21 так і "
                         "не звелось, і дрон зачепив косяк гвинтом")
    ap.add_argument("--trust-normal", action="store_true",
                    help="довіряти орієнтації обруча з розкладу коніки. "
                         "ВИМКНЕНО за замовчуванням: еліпс будується з "
                         "bbox YOLO і не має орієнтації, тому нормаль "
                         "систематично хибна (політ 01.10: обруч стояв "
                         "вертикально, нормаль виходила 'лежить під "
                         "45°', і дрон летів убік усіма каналами в "
                         "насиченні). Дистанція з bbox надійна, тому "
                         "PBVS лишається корисним без орієнтації")
    ap.add_argument("--vertical-bias", type=float, default=-0.20,
                    help="сталий зсув цілі по вертикалі в кадрі. "
                         "Камера Tello дивиться вперед і трохи вниз, "
                         "тому обруч на своїй висоті видно ВИЩЕ за "
                         "центр. Без поправки умова центрування не "
                         "виконується ніколи: політ 01.10 дав медіану "
                         "|err_y| = 0.209 при допуску 0.08, і дрон "
                         "26 с ліз угору замість зближення")
    ap.add_argument("--speed", type=float, default=0.35,
                    help="стеля швидкості контролера, частка каналу")
    ap.add_argument("--near-throttle", type=float, default=0.9,
                    help="частка швидкості на близькій дистанції "
                         "(fill > approach_fill)")
    ap.add_argument("--align-throttle", type=float, default=0.6,
                    help="частка швидкості у фазі ALIGN")

    ap.add_argument("--blind-seconds", type=float, default=2.0,
                    help="скільки летіти вперед, коли обруч перестав "
                         "вміщатися в кадр і поза вже не відновлюється")
    ap.add_argument("--lost-commit", type=float, default=2.5,
                    help="якщо ціль зникла ближче за цю дистанцію — це "
                         "не втрата, а вхід у кільце: йдемо наосліп")

    ap.add_argument("--compensate", choices=["none", "linear", "kalman"],
                    default="kalman",
                    help="компенсація затримки контуру")
    ap.add_argument("--fixed-latency", type=float, default=None,
                    help="затримка в секундах. Без неї ОЦІНЮЄТЬСЯ на "
                         "льоту з телеметрії")

    ap.add_argument("--view", action="store_true", help="вікно з відео")
    ap.add_argument("--record", action="store_true", help="писати відео й лог")
    ap.add_argument("--out", default="experiments/results/tello_fly")
    return fly(ap.parse_args())


def build_controller(args, w: int, h: int):
    """Контролер за `--strategy`. Компенсація — однакова для всіх."""
    fx = args.fx if args.fx else (w / 2.0) / np.tan(np.deg2rad(82.0) / 2.0)
    K = np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])

    def kw():
        return {"predictor": make_predictor(args.compensate, 2),
                "latency": args.fixed_latency or 0.05,
                "max_speed": args.speed}

    def pbvs_kw():
        bias_m = args.vertical_bias * args.switch_distance * (h / 2.0) / fx
        return dict(kw(), trust_normal=args.trust_normal,
                    vertical_bias=bias_m,
                    min_lateral=args.min_lateral_channel
                    / max(1.0, args.max_channel))

    def ibvs_kw():
        return dict(kw(), vertical_bias=args.vertical_bias,
                    near_throttle=args.near_throttle,
                    align_throttle=args.align_throttle,
                    min_lateral=args.min_lateral_channel
                    / max(1.0, args.max_channel))

    if args.strategy == "ibvs":
        return IBVSController(**ibvs_kw()), K
    if args.strategy == "pbvs":
        return PBVSController(PnPSolver(K, target_diameter=args.diameter),
                              **pbvs_kw()), K
    return HybridController(
        PBVSController(PnPSolver(K, target_diameter=args.diameter),
                       **pbvs_kw()),
        IBVSController(**ibvs_kw()),
        switch_distance=args.switch_distance,
        max_speed=args.speed), K


def fly(args) -> int:
    from ultralytics import YOLO

    need_vision = args.mode != "hover"
    model = None
    if need_vision:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA недоступна -> cpu")
            args.device = "cpu"
        dev = args.device
        model = YOLO(args.weights)
        model.to(dev)
        print(f"детектор: {args.weights} на {dev}")

    link = TelloLink(max_channel=args.max_channel)
    print("під'єднуюся ...")
    tel = link.connect(stream=need_vision)
    print(f"батарея {tel.battery}% | висота {tel.height_cm} см | "
          f"tof {tel.tof_cm} см | темп {tel.temp_c:.0f}°C | "
          f"крен {tel.roll}° тангаж {tel.pitch}° курс {tel.yaw}°")
    if tel.temp_c >= TEMP_LIMIT:
        print(f"температура {tel.temp_c:.0f}°C >= {TEMP_LIMIT} — не злітаю")
        return 1

    w = h = 0
    if need_vision:
        t_wait = time.time()
        while time.time() - t_wait < 12.0:
            frame = link.frame()
            if frame is not None and frame.shape[:2] != STUB_SHAPE:
                h, w = frame.shape[:2]
                break
            print("no frame!")
            time.sleep(0.4)
        if not w:
            print("відео не піднялося")
            link.close()
            return 1
        print(f"кадр: {w}x{h}")
        model.predict(np.zeros((h, w, 3), np.uint8), verbose=False,
                      device=args.device)

    ctrl = pose_solver = None
    if args.mode == "approach":
        ctrl, K = build_controller(args, w, h)
        pose_solver = PnPSolver(K, target_diameter=args.diameter)
        print(f"контролер: {args.strategy}", end="")
        if args.strategy == "hybrid":
            print(f" (PBVS > {args.switch_distance} м, IBVS ближче)", end="")
        print(f" | компенсація: {args.compensate}", end="")
        print(f", затримка {'фіксована ' + str(args.fixed_latency)
                            if args.fixed_latency else 'оцінюється на льоту'}")

    track_pid = PID(1.2, 0.0, 0.15)
    lat_est = LatencyEstimator(fallback=args.fixed_latency or 0.05)

    stop = {"flag": False}

    def on_sigint(*_):
        print("\nперервано -> сідаю")
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_sigint)

    out_dir = Path(args.out)
    writer = None
    rows: list[dict] = []
    win = (LiveWindow(WIN, w, h).start()
           if (args.view and need_vision) else None)

    print(f"\nЗЛІТ — режим {args.mode}, {args.seconds:g} с, "
          f"канал <= {args.max_channel}")
    link.takeoff()
    if win is not None:
        print("вікно відкрито — Q або ESC садить дрон")

    t0 = last = time.time()
    n = starved = no_target = 0
    fps_now = 0.0
    t_fps, n_fps = time.time(), 0
    blind_left = 0.0
    passed = False
    last_dist = None
    last_seen = 0.0
    tel_live = tel
    try:
        while time.time() - t0 < args.seconds and not stop["flag"]:
            now = time.time()
            dt = max(1e-3, now - last)
            last = now
            n += 1

            tel_live = link.telemetry()

            if args.mode == "hover":
                link.hover()
                time.sleep(0.05)
                continue

            frame = link.frame()
            if frame is None or frame.shape[:2] == STUB_SHAPE:
                link.watchdog()
                starved += 1
                continue

            res = model.predict(frame, verbose=False, conf=0.25,
                                device=args.device)[0]
            det = build_detection(res, frame.shape)

            n_fps += 1
            if time.time() - t_fps >= 0.5:
                fps_now = n_fps / (time.time() - t_fps)
                t_fps, n_fps = time.time(), 0

            cmd = None
            rc = (0, 0, 0, 0)
            pose = None

            if det is not None and det.confidence >= args.min_conf:
                no_target = 0
                last_seen = now
                pose = pose_solver.solve(det) if pose_solver else None
                if pose is not None:
                    last_dist = float(pose.distance)
                lat_est.update(now, float(tel_live.yaw), det.center[0])
                if ctrl is not None and args.fixed_latency is None:
                    _set_latency(ctrl, lat_est.latency)

                if args.mode == "track":
                    cmd = track_yaw(det, w, track_pid, dt)
                else:
                    cmd = ctrl.update(det, (h, w), dt)
                rc = link.send(cmd)

            elif args.mode == "approach" and blind_left > 0.0:
                blind_left -= dt
                if blind_left <= 0.0:
                    passed = True
                    print(f"  {now - t0:5.1f}s  ПРОЛІТ ЗАВЕРШЕНО "
                          f"(сліпий шлях {args.blind_seconds:.1f} с "
                          f"пройдено)")
                    link.hover()
                    cmd = ControlCommand(phase="PASSED")
                    rc = (0, 0, 0, 0)
                else:
                    cmd = ControlCommand(forward=args.speed,
                                         phase=f"BLIND({blind_left:.1f}с)")
                    rc = link.send(cmd)

            elif passed:
                link.hover()
                cmd = ControlCommand(phase="PASSED")

            else:
                no_target += 1
                if (args.mode == "approach" and blind_left <= 0.0
                        and last_dist is not None
                        and last_dist <= args.lost_commit
                        and now - last_seen < 1.0):
                    blind_left = args.blind_seconds
                    print(f"  {now - t0:5.1f}s  ЦІЛЬ ЗНИКЛА на "
                          f"{last_dist:.2f} м -> сліпо "
                          f"{args.blind_seconds:.1f} с")
                else:
                    link.hover()
                    if no_target % 20 == 0:
                        print(f"  {now - t0:5.1f}s  ЦІЛІ НЕ ВИДНО "
                              f"({no_target} кадрів)")

            if win is not None or args.record:
                vis = draw_overlay(frame, det, pose, cmd, rc, tel_live,
                                   fps_now, args.mode)
                if win is not None:
                    win.show(vis)
                    if win.quit_requested:
                        stop["flag"] = True
                if args.record:
                    if writer is None:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(out_dir / "flight.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
                    writer.write(vis)

            if args.record and cmd is not None:
                rows.append({
                    "t": round(now - t0, 3),
                    "det": int(det is not None),
                    "conf": round(det.confidence, 3) if det else None,
                    "dist": round(pose.distance, 3) if pose else None,
                    "phase": cmd.phase,
                    "lat": rc[0], "fwd": rc[1], "vert": rc[2], "yaw": rc[3],
                    "lat_ms": round(lat_est.latency * 1000, 1),
                    "lat_q": round(lat_est.quality, 2),
                })

            if n % 30 == 0 and cmd is not None:
                d = f" d={pose.distance:.2f}м" if pose else ""
                print(f"  {now - t0:5.1f}s  {cmd.phase[:40]}{d}  rc={rc}")
    finally:
        print("\nсідаю ...")
        link.land()
        tel_end = link.telemetry()
        print(f"батарея {tel_end.battery}% | темп {tel_end.temp_c:.0f}°C | "
              f"курс {tel_end.yaw}°")
        det_n = sum(r["det"] for r in rows) if rows else 0
        print(f"кадрів: {n}, з детекцією: {det_n}"
              f"{f' ({100 * det_n // max(1, len(rows))}%)' if rows else ''}")
        print(f"ітерацій без кадру: {starved}")
        print(f"проліт: {'ЗАРАХОВАНО' if passed else 'не зараховано'}")
        if lat_est.updates:
            print(f"затримка (оцінка на льоту): "
                  f"{lat_est.latency * 1000:.0f} мс, "
                  f"якість {lat_est.quality:.2f}, "
                  f"оновлень {lat_est.updates}")
        if writer is not None:
            writer.release()
            print(f"відео: {out_dir / 'flight.mp4'}")
        if rows:
            out_dir.mkdir(parents=True, exist_ok=True)
            keys = sorted({k for r in rows for k in r})
            with open(out_dir / "flight_log.csv", "w", newline="",
                      encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=keys)
                wr.writeheader()
                wr.writerows(rows)
            print(f"лог: {out_dir / 'flight_log.csv'}")
        if win is not None:
            win.close()
        link.close()
    return 0


def _set_latency(ctrl, value: float) -> None:
    """Роздати оцінку затримки контролеру й усім його підстратегіям."""
    for obj in (ctrl, getattr(ctrl, "pbvs", None), getattr(ctrl, "ibvs", None)):
        if obj is not None and hasattr(obj, "latency"):
            obj.latency = value


if __name__ == "__main__":
    raise SystemExit(main())
