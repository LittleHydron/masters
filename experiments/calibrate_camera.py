"""Калібрування камери за відео з шаховим патерном.

Потрібне, бо PBVS відновлює МЕТРИЧНУ позу: похибка фокусної відстані
переходить прямо в похибку дистанції. IBVS до калібрування байдужий —
і саме цю різницю в чутливості показує порівняння стратегій.

Кадри беруться з відео з проріджуванням: сусідні кадри майже однакові
й нічого не додають, зате роздувають час обчислень.

Приклад:
    python -m experiments.calibrate_camera --video calibration/calib.mov \\
        --square 28.0 --name iphone14
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--pattern", nargs=2, type=int, default=[8, 5],
                    metavar=("COLS", "ROWS"),
                    help="ВНУТРІШНІ кути (для дошки 9x6 це 8x5)")
    ap.add_argument("--square", type=float, required=True,
                    help="реальна сторона клітинки в мм — ВИМІРЯНА лінійкою")
    ap.add_argument("--every", type=int, default=15,
                    help="брати кожен N-й кадр")
    ap.add_argument("--max-views", type=int, default=40)
    ap.add_argument("--name", default="camera")
    ap.add_argument("--out", default="calibration")
    args = ap.parse_args()

    cols, rows = args.pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
                   * args.square).astype(np.float32)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Не відкрити {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Відео: {w}x{h}, кадрів {total}")

    obj_points, img_points = [], []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    idx = checked = 0

    while len(obj_points) < args.max_views:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.every:
            continue
        checked += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_points.append(objp.copy())
        img_points.append(corners)
        print(f"  кадр {idx}: дошку знайдено ({len(obj_points)})")

    cap.release()
    print(f"\nПеревірено кадрів: {checked}, придатних ракурсів: {len(obj_points)}")

    if len(obj_points) < 8:
        raise SystemExit(
            f"Замало ракурсів ({len(obj_points)}). Потрібно >=8, краще 15-25.\n"
            "Знімайте дошку під РІЗНИМИ кутами й на різних відстанях, "
            "заповнюючи різні частини кадру."
        )

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None)

    errs = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        a = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(img_points[i], dtype=np.float64).reshape(-1, 2)
        errs.append(float(np.sqrt(((a - b) ** 2).sum(axis=1)).mean()))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    fov_x = float(np.degrees(2 * np.arctan(w / (2 * fx))))
    fov_y = float(np.degrees(2 * np.arctan(h / (2 * fy))))

    print(f"\nRMS перепроєкції: {rms:.3f} px  (добре < 0.5, прийнятно < 1.0)")
    print(f"  по ракурсах: медіана {np.median(errs):.3f}, макс {max(errs):.3f}")
    print(f"\nfx={fx:.1f}  fy={fy:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"Поле зору: {fov_x:.1f}° x {fov_y:.1f}°")
    print(f"Дисторсія k1={dist[0][0]:+.4f} k2={dist[0][1]:+.4f}")

    out = Path(args.out) / f"{args.name}_{w}x{h}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "name": args.name, "width": w, "height": h,
        "K": K.tolist(), "dist": dist.tolist(),
        "fov_x_deg": round(fov_x, 2), "fov_y_deg": round(fov_y, 2),
        "rms_px": round(float(rms), 4),
        "views": len(obj_points), "square_mm": args.square,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nЗбережено: {out}")

    if rms > 1.0:
        print("\nУВАГА: RMS завеликий. Ймовірні причини — патерн не плоский "
              "(приклейте до картону), замало ракурсів, або неточно виміряна "
              "клітинка.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
