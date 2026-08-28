"""Показати шаховий патерн на екрані для калібрування без друку.

Розмір клітинки задається в ПІКСЕЛЯХ і перераховується в міліметри
за фізичним розміром екрана (xrandr). Це точніше за друк: принтери
масштабують, а крок пікселя монітора фіксований.

Екран мусить бути ПЛОСКИМ (не вигнутим) і чистим. Яскравість —
середню: пересвітлений екран змиває кути.

Приклад:
    python -m experiments.show_calib_pattern --square-px 172 \\
        --screen-mm 800 --screen-px 3440
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-px", type=int, required=True)
    ap.add_argument("--screen-px", type=int, required=True,
                    help="ширина екрана в пікселях (з xrandr)")
    ap.add_argument("--screen-mm", type=float, required=True,
                    help="ширина екрана в мм (з xrandr)")
    ap.add_argument("--save", default="",
                    help="зберегти PNG замість показу")
    args = ap.parse_args()

    pitch = args.screen_mm / args.screen_px
    square_mm = args.square_px * pitch

    sq = args.square_px
    bw, bh = args.cols * sq, args.rows * sq
    # Біле поле навколо дошки — findChessboardCorners його вимагає.
    pad = sq
    img = np.full((bh + 2 * pad, bw + 2 * pad), 255, np.uint8)
    for r in range(args.rows):
        for c in range(args.cols):
            if (r + c) % 2:
                y0, x0 = pad + r * sq, pad + c * sq
                img[y0:y0 + sq, x0:x0 + sq] = 0

    print(f"Крок пікселя: {pitch:.4f} мм")
    print(f"КЛІТИНКА = {square_mm:.2f} мм   <- це значення для --square")
    print(f"Внутрішніх кутів: {args.cols - 1}x{args.rows - 1}  <- для --pattern")
    print(f"Розмір патерну на екрані: {bw * pitch:.0f} x {bh * pitch:.0f} мм")

    if args.save:
        cv2.imwrite(args.save, img)
        print(f"\nЗбережено: {args.save}")
        return 0

    win = "calib"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, img)
    print("\nПоказано на весь екран. Знімайте з різних ракурсів.")
    print("Вихід — клавіша q або Esc у вікні патерну.")
    while True:
        k = cv2.waitKey(100) & 0xFF
        if k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
