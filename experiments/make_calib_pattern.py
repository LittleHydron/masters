"""Згенерувати шаховий патерн для калібрування камери у PDF (A4).

Друкувати БЕЗ масштабування («Actual size» / «100%»), інакше реальний
розмір клітинки не збігатиметься із заявленим і фокусна відстань
вийде зміщеною.

Після друку ОБОВ'ЯЗКОВО виміряти сторону клітинки лінійкою і
передати виміряне значення в `calibrate_camera.py --square`.

Приклад:
    python -m experiments.make_calib_pattern
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

A4_W_MM, A4_H_MM = 210.0, 297.0
DPI = 300


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=10,
                    help="клітинок по горизонталі (внутрішніх кутів буде cols-1)")
    ap.add_argument("--rows", type=int, default=7)
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--out", default="calibration/pattern_a4.pdf")
    args = ap.parse_args()

    sq = mm_to_px(args.square_mm)
    board_w, board_h = args.cols * sq, args.rows * sq
    page_w, page_h = mm_to_px(A4_W_MM), mm_to_px(A4_H_MM)

    margin = mm_to_px(10.0)
    if board_w > page_w - margin and board_w <= page_h - margin:
        page_w, page_h = page_h, page_w
        landscape = True
    else:
        landscape = False

    if board_w > page_w - margin or board_h > page_h - margin:
        raise SystemExit(
            f"Дошка {args.cols}x{args.rows} по {args.square_mm} мм не влазить в A4. "
            f"Зменшіть --square-mm або кількість клітинок."
        )

    page = np.full((page_h, page_w), 255, np.uint8)
    ox, oy = (page_w - board_w) // 2, (page_h - board_h) // 2
    for r in range(args.rows):
        for c in range(args.cols):
            if (r + c) % 2 == 0:
                continue
            y0, x0 = oy + r * sq, ox + c * sq
            page[y0:y0 + sq, x0:x0 + sq] = 0

    txt = (f"{args.cols}x{args.rows} squares, {args.square_mm:.0f} mm  |  "
           f"inner corners {args.cols - 1}x{args.rows - 1}  |  "
           f"PRINT AT 100%, THEN MEASURE")
    cv2.putText(page, txt, (ox, oy + board_h + mm_to_px(8)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    cv2.imwrite(str(png), page)

    try:
        from PIL import Image
        Image.open(png).save(out, "PDF", resolution=DPI)
        made = out
    except ImportError:
        made = png
        print("PIL не встановлено — збережено лише PNG")

    print(f"Патерн: {made}  ({'альбомна' if landscape else 'книжкова'} орієнтація)")
    print(f"  клітинок: {args.cols}x{args.rows}, сторона {args.square_mm:.0f} мм")
    print(f"  внутрішніх кутів: {args.cols - 1}x{args.rows - 1}  <- це для --pattern")
    print("\nДРУКУВАТИ зі 100% масштабом, потім ВИМІРЯТИ клітинку лінійкою.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
