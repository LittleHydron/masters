"""Перекодувати відео в H.264, щоб воно відтворювалось усюди.

OpenCV на цій машині вміє писати лише FMP4 (`mp4v`): апаратного
кодувальника H.264 немає, а програмного в збірці не зібрано. FMP4
відкривається не всіма плеєрами — для показу комісії це не годиться.

ffmpeg береться з пакета `imageio-ffmpeg` (власний статичний бінарник,
без потреби в системному пакеті чи правах root).

Приклад:
    python -m experiments.to_h264 experiments/results_real/*.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def convert(src: Path, crf: int, scale: int | None) -> Path | None:
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    dst = src.with_name(src.stem + "_h264.mp4")
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src)]
    if scale:
        cmd += ["-vf", f"scale={scale}:-2"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            # yuv420p — інакше частина плеєрів і браузерів не покаже
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"  ПОМИЛКА {src.name}: {r.stderr.strip()[:200]}")
        return None
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--crf", type=int, default=26,
                    help="якість: менше = краще й важче (18-28)")
    ap.add_argument("--scale", type=int, default=0,
                    help="ширина в пікселях, 0 = не міняти")
    ap.add_argument("--replace", action="store_true",
                    help="видалити вихідний FMP4 після успішного "
                         "перекодування (він у 10-15 разів більший)")
    args = ap.parse_args()

    for v in args.videos:
        src = Path(v)
        if not src.exists() or src.stem.endswith("_h264"):
            continue
        dst = convert(src, args.crf, args.scale or None)
        if dst:
            a, b = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
            print(f"  {src.name} -> {dst.name}  {a:.0f} -> {b:.0f} МБ")
            if args.replace:
                src.unlink()
                print(f"    видалено {src.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
