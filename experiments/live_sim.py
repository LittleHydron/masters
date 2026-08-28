"""Інтерактивний симулятор: вікно з польотом у реальному часі.

Показує одночасно два види:
* **сцена збоку** — умовний вигляд «з боку»: обруч, дрон, траєкторія;
* **камера дрона** — те, що бачить бортова камера, з накладеною
  детекцією YOLO, оціненою позою та командами трьох контролерів.

Контур замкнений і всі ланки справжні, крім спрощеної фізики руху й
процедурного фону: рендер кадру -> YOLO -> розклад коніки -> контролер
-> рух дрона -> наступний кадр.

Керування у вікні:
    ПРОБІЛ  пауза / продовжити
    R       перезапустити політ
    1/2/3   керує IBVS / PBVS / гібрид
    S       зберегти поточний кадр
    Q, Esc  вихід

Приклад:
    python -m experiments.live_sim --weights models/yolov8n_mixed2.pt
"""
from __future__ import annotations

import argparse
import sys
import time
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

COLORS = {"ibvs": (90, 200, 255), "pbvs": (140, 255, 140),
          "hybrid": (255, 170, 120)}
ORDER = ["ibvs", "pbvs", "hybrid"]


def intrinsics(w: int, h: int, fov_deg: float) -> np.ndarray:
    fx = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])


def draw_scene(w: int, h: int, cam: np.ndarray, trail: list,
               hoop_normal: np.ndarray, radius: float,
               status: str) -> np.ndarray:
    """Вигляд ЗГОРИ на сцену: обруч, дрон, пройдений шлях.

    Площина XZ: горизонталь — бічне зміщення, вертикаль — дальність.
    Раніше підпис казав «вид збоку», і глядач шукав висоту там, де
    насправді показана дальність.
    """
    img = np.full((h, w, 3), 22, np.uint8)
    # Шрифти й товщини масштабуємо від висоти панелі, щоб скріншот
    # лишався читабельним і на 480, і на 900 px.
    k = h / 480.0
    x0, z0, x1, z1 = -3.0, -6.0, 3.0, 1.5

    def px(p):
        x = int((p[0] - x0) / (x1 - x0) * (w - 40) + 20)
        y = int(h - 40 - (p[2] - z0) / (z1 - z0) * (h - 80))
        return x, y

    # Сітка з підписами метрів — щоб було видно масштаб.
    for zz in range(-6, 2):
        y = px(np.array([0, 0, zz]))[1]
        cv2.line(img, (18, y), (w - 18, y), (40, 40, 40), 1)
        cv2.putText(img, f"{zz}m", (int(6 * k), y + int(5 * k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * k, (95, 95, 95),
                    max(1, int(k)))
    for xx in (-2, -1, 0, 1, 2):
        x = px(np.array([xx, 0, 0]))[0]
        cv2.line(img, (x, 20), (x, h - 24), (40, 40, 40), 1)

    # Обруч — відрізок, перпендикулярний нормалі.
    perp = np.array([-hoop_normal[2], 0.0, hoop_normal[0]])
    n = np.linalg.norm(perp)
    if n > 1e-6:
        perp = perp / n * radius
        a, b = px(-perp), px(perp)
        cv2.line(img, a, b, (60, 90, 240), max(5, int(7 * k)))
        tip = px(hoop_normal * 0.55)
        cv2.arrowedLine(img, px(np.zeros(3)), tip, (70, 110, 220),
                        max(2, int(3 * k)), tipLength=0.3)
    cv2.putText(img, "HOOP", (px(np.zeros(3))[0] - int(40 * k),
                              px(np.zeros(3))[1] - int(20 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58 * k, (110, 140, 250),
                max(1, int(1.6 * k)))

    for p, q in zip(trail, trail[1:]):
        cv2.line(img, px(p), px(q), (80, 170, 230), max(2, int(2.5 * k)))

    # Дрон.
    c = px(cam)
    cv2.circle(img, c, int(10 * k), (255, 255, 255), -1)
    cv2.circle(img, c, int(16 * k), (200, 200, 200), max(1, int(k)))
    for dx in (int(-18 * k), int(18 * k)):
        cv2.line(img, (c[0] + dx, c[1] - int(9 * k)),
                 (c[0] + dx, c[1] + int(9 * k)), (170, 170, 170),
                 max(2, int(2.5 * k)))
    cv2.putText(img, "DRONE", (c[0] - int(26 * k), c[1] + int(38 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, (225, 225, 225),
                max(1, int(1.5 * k)))

    cv2.putText(img, "SCENE (top view)", (int(18 * k), int(30 * k)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65 * k, (165, 165, 165),
                max(1, int(1.4 * k)))
    if status:
        col = (90, 230, 120) if status == "PASSED" else (80, 80, 240)
        cv2.putText(img, status, (w // 2 - int(95 * k), h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8 * k, col, max(3, int(4 * k)))
    return img


def draw_arrow(img, cx, cy, lat, fwd, vert, color, scale=44):
    dx, dy = lat * scale, -vert * scale
    th = max(2, int(2 + abs(fwd) * 7))
    if abs(dx) < 2 and abs(dy) < 2:
        cv2.circle(img, (int(cx), int(cy)), 7, color, th)
    else:
        cv2.arrowedLine(img, (int(cx), int(cy)),
                        (int(cx + dx), int(cy + dy)), color, th, tipLength=0.3)
    if fwd > 0.05:
        cv2.circle(img, (int(cx), int(cy)), int(6 + fwd * 11), color, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/yolov8n_mixed2.pt")
    ap.add_argument("--controller", default="hybrid", choices=ORDER)
    ap.add_argument("--diameter", type=float, default=0.89)
    ap.add_argument("--width", type=int, default=900,
                    help="ширина кадру камери (більше = читабельніші скріни)")
    ap.add_argument("--height", type=int, default=675)
    ap.add_argument("--fov", type=float, default=82.0)
    ap.add_argument("--start", nargs=3, type=float, default=[1.1, 0.35, -4.5])
    ap.add_argument("--tilt", type=float, default=18.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--latency", type=float, default=0.20)
    ap.add_argument("--fps", type=float, default=15.0, help="цільовий темп показу")
    ap.add_argument("--seed", type=int, default=-1,
                    help="-1 = випадково при кожному запуску й рестарті")
    ap.add_argument("--fixed", action="store_true",
                    help="однакова сцена щоразу (для відтворюваності)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    w, h = args.width, args.height
    K = intrinsics(w, h, args.fov)
    radius = args.diameter / 2.0
    spec = HoopSpec(diameter=args.diameter, tube_diameter=0.020,
                    color_bgr=(40, 40, 200))

    scene_w, scene_h = int(h * 1.05), h
    panel_w = int(h * 0.62)
    active = args.controller

    def build():
        """Нова сцена: фон, нахил обруча й стартова позиція.

        Кожен рестарт (клавіша R) дає ІНШИЙ політ — інакше демонстрація
        показує лише один вивчений випадок.
        """
        seed = args.seed if args.seed >= 0 else int(time.time() * 1000) % 2**31
        rng = np.random.default_rng(seed)
        bg = procedural_room(w, h, rng, distractor_circles=False).astype(np.uint8)

        if args.fixed:
            tilt_deg = args.tilt
            start = np.array(args.start, dtype=float)
            t = np.deg2rad(tilt_deg)
            normal = np.array([np.sin(t), 0.0, np.cos(t)])
        else:
            # Еліптичність обруча в кадрі задає кут між його нормаллю
            # та ОПТИЧНОЮ ВІССЮ камери (тут вона завжди +z), а не
            # напрямком на ціль: камера в цій моделі не повертається,
            # тож бічне зміщення саме по собі обруч не «сплющує».
            #
            # Раніше я рахував кут до напрямку на обруч — і виходила
            # нісенітниця: при нахилі 0° «кут огляду 26°», а проєкція
            # ідеальне коло (a/b = 1.00). Тепер нахил ЗАДАЄ еліптичність
            # напряму: a/b = cos(tilt).
            start = np.array([rng.uniform(-1.6, 1.6),
                              rng.uniform(-0.6, 0.6),
                              -rng.uniform(2.2, 3.6)])
            # Виміряно: при 25° сплющення практично невидиме
            # (a/b = 0.95), при 40° a/b = 0.83, при 55° a/b = 0.66.
            # Перспектива послаблює ефект, тому беремо 30-58°.
            tilt_deg = float(30 + 28 * rng.beta(1.8, 1.2))
            if rng.random() < 0.5:
                tilt_deg = -tilt_deg
            t = np.deg2rad(tilt_deg)
            normal = np.array([np.sin(t), 0.0, np.cos(t)])
        ctrls = {
            "ibvs": IBVSController(),
            "pbvs": PBVSController(PnPSolver(K, target_diameter=args.diameter)),
            "hybrid": HybridController(
                PBVSController(PnPSolver(K, target_diameter=args.diameter)),
                IBVSController(), switch_distance=2.0),
        }
        drone = DroneModel(start, DroneParams(max_speed=args.speed,
                                             latency=args.latency), rng)
        # Кут до оптичної осі (+z) — саме він визначає еліптичність.
        got = np.degrees(np.arccos(np.clip(abs(float(normal[2])), 0, 1)))
        print(f"  нова сцена: нахил {got:.0f}° (a/b={np.cos(np.deg2rad(got)):.2f}), "
              f"старт [{start[0]:+.2f} {start[1]:+.2f} {start[2]:+.2f}], seed {seed}")
        return rng, bg, ctrls, drone, normal

    rng, background, ctrls, drone, hoop_normal = build()
    solver = PnPSolver(K, target_diameter=args.diameter)
    trail: list[np.ndarray] = []
    prev_side = float(drone.p @ hoop_normal)
    status = ""
    step = 0
    paused = False
    dt = 1 / 30.0
    win = "БПЛА: симуляція прольоту крізь обруч"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w + scene_w + panel_w, h)

    print(f"Модель: {args.weights}")
    print("ПРОБІЛ пауза | R перезапуск | 1/2/3 контролер | S кадр | Q вихід")

    while True:
        frame_start = time.perf_counter()
        if not paused and not status:
            cam = drone.p.copy()
            trail.append(cam.copy())
            # Обруч у нулі; враховуємо і позицію, і КУРС дрона.
            # Без повороту картинка не відповідала б стану дрона:
            # він обертається під час обльоту, а камера дивилась би
            # завжди вздовж +z, і контролер отримував би хибний ракурс.
            rel = drone.to_camera(np.zeros(3))
            rel_n = drone.rotate_to_camera(hoop_normal)
            res = render_hoop(background, K, rel, rel_n, spec, rng=rng)
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
            pose = solver.solve(det) if det is not None else None
            cmds = {n: c.update(det, (h, w), dt) for n, c in ctrls.items()}

            drone.step(cmds[active], dt)
            side = float(drone.p @ hoop_normal)
            if prev_side < 0 <= side:
                a = prev_side / (prev_side - side) if prev_side != side else 0.0
                hit = cam + (drone.p - cam) * a
                r = float(np.linalg.norm(hit - hoop_normal * float(hit @ hoop_normal)))
                status = "PASSED" if r < radius else "MISSED"
                print(f"  крок {step}: {status} (край {r - radius:+.2f} м)")
            prev_side = side
            if np.linalg.norm(drone.p) > 12.0:
                status = "LOST"
            step += 1
            last = (frame, det, pose, cmds, cam)
        elif "last" in dir():
            frame, det, pose, cmds, cam = last
        else:
            continue

        # --- камера дрона ---
        kk = h / 480.0
        view = frame.copy()
        if det is not None:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 230, 0),
                          max(2, int(3 * kk)))
            cv2.putText(view, f"{det.confidence:.2f}",
                        (x1, max(int(24 * kk), y1 - int(8 * kk))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75 * kk, (0, 230, 0),
                        max(2, int(2 * kk)))
        cv2.drawMarker(view, (w // 2, h // 2), (110, 110, 110),
                       cv2.MARKER_TILTED_CROSS, int(18 * kk), max(1, int(kk)))
        cv2.putText(view, "DRONE CAMERA", (int(14 * kk), int(30 * kk)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * kk, (180, 180, 180),
                    max(1, int(1.4 * kk)))

        # --- панель ---
        panel = np.full((h, panel_w, 3), 18, np.uint8)
        F = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, f"frame {step}", (int(18 * kk), int(38 * kk)),
                    F, 0.65 * kk, (175, 175, 175), max(1, int(1.3 * kk)))
        cv2.putText(panel, f"true dist = {np.linalg.norm(cam):.2f} m",
                    (int(18 * kk), int(72 * kk)), F, 0.6 * kk,
                    (145, 145, 145), max(1, int(1.2 * kk)))
        if pose is not None:
            tilt = np.degrees(np.arccos(np.clip(abs(pose.normal[2]), 0, 1)))
            cv2.putText(panel, f"d = {pose.distance:.2f} m",
                        (int(18 * kk), int(124 * kk)), F, 1.05 * kk,
                        (0, 210, 255), max(2, int(2.6 * kk)))
            cv2.putText(panel, f"tilt = {tilt:.0f} deg",
                        (int(18 * kk), int(158 * kk)), F, 0.65 * kk,
                        (0, 210, 255), max(1, int(1.4 * kk)))
        else:
            cv2.putText(panel, "NO TARGET", (int(18 * kk), int(124 * kk)),
                        F, 1.0 * kk, (70, 70, 235), max(2, int(2.6 * kk)))

        # Крок блоку рахуємо від висоти: інакше на низькому вікні
        # третій контролер налазить на підказки внизу.
        y0 = int(h * 0.38)
        # Блок = заголовок + фаза + команди (58*kk) плюс проміжок,
        # інакше назва наступного контролера налазить на попередні
        # команди — так було на першій версії великого вікна.
        step_y = max(int(88 * kk), int((h - y0 - int(80 * kk)) / 3))
        y = y0
        for name in ORDER:
            c = cmds[name]
            col = COLORS[name]
            on = (name == active)
            cv2.line(panel, (int(7 * kk), y - int(24 * kk)),
                     (int(7 * kk), y + int(56 * kk)), col,
                     int(6 * kk) if on else int(3 * kk))
            cv2.putText(panel, name.upper() + ("  <=" if on else ""),
                        (int(20 * kk), y), F, 0.78 * kk, col,
                        max(2, int(2.2 * kk)) if on else max(1, int(1.4 * kk)))
            cv2.putText(panel, c.phase[:20], (int(20 * kk), y + int(30 * kk)),
                        F, 0.56 * kk, (180, 180, 180), max(1, int(1.2 * kk)))
            cv2.putText(panel,
                        f"L{c.lateral:+.2f} F{c.forward:+.2f} V{c.vertical:+.2f}",
                        (int(20 * kk), y + int(58 * kk)), F, 0.56 * kk, col,
                        max(1, int(1.3 * kk)))
            draw_arrow(panel, panel_w - int(58 * kk), y + int(16 * kk),
                       c.lateral, c.forward, c.vertical, col,
                       scale=int(52 * kk))
            y += step_y
        cv2.putText(panel, "SPACE pause   R restart",
                    (int(16 * kk), h - int(44 * kk)), F, 0.5 * kk,
                    (120, 120, 120), max(1, int(1.2 * kk)))
        cv2.putText(panel, "1/2/3 controller   S shot   Q quit",
                    (int(16 * kk), h - int(20 * kk)), F, 0.5 * kk,
                    (120, 120, 120), max(1, int(1.2 * kk)))

        scene = draw_scene(scene_w, scene_h, cam, trail, hoop_normal,
                           radius, status)
        cv2.imshow(win, np.hstack([scene, view, panel]))

        # Темп показу.
        delay = max(1, int((1.0 / args.fps - (time.perf_counter() - frame_start)) * 1000))
        key = cv2.waitKey(delay if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            rng, background, ctrls, drone, hoop_normal = build()
            trail.clear(); prev_side = float(drone.p @ hoop_normal)
            status = ""; step = 0
        if key in (ord("1"), ord("2"), ord("3")):
            active = ORDER[key - ord("1")]
            print(f"  керує: {active.upper()}")
        if key == ord("s"):
            out = Path("experiments/results_demo")
            out.mkdir(parents=True, exist_ok=True)
            p = out / f"live_{step:05d}.jpg"
            cv2.imwrite(str(p), np.hstack([scene, view, panel]))
            print(f"  збережено {p}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
