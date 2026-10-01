"""Зведення ВСІХ метрик роботи в одну таблицю.

Метрики розкидані по семи файлах, отриманих у різні дні різними
скриптами. Для тексту роботи, захисту й публікації потрібна одна
таблиця, де видно: що міряли, чим, на яких даних і коли.

Скрипт нічого не рахує заново — лише збирає готові результати й
позначає, яких із них бракує. Джерело кожного числа вказане поруч,
щоб на захисті можна було показати, звідки воно взялося.

Приклад:
    python -m experiments.collect_metrics
    python -m experiments.collect_metrics --md > METRICS.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    if not path.exists():
        return None
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text())
        return list(csv.DictReader(open(path)))
    except Exception:
        return None


def _stamp(path: Path) -> str:
    if not path.exists():
        return "—"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%d.%m.%Y")


def detection_metrics() -> list[dict]:
    """Задача 1: детекція на розмічених кадрах, зведено по 4 сценах.

    Агрегуємо СУМОЮ TP/FP/FN, а не середнім із посценових precision:
    середнє з відношень дає інше число, ніж відношення сум, і на
    сценах різного розміру це помітно спотворює результат.

    F0.5 важить precision удвічі проти recall. Для дрона хибне
    спрацювання дорожче за пропуск: краще не побачити обруч і
    зависнути, ніж полетіти в стіну, прийнявши її за ціль.
    """
    p = ROOT / "experiments/results/labeled_eval.json"
    d = _load(p)
    if not isinstance(d, dict):
        return []
    rows = []
    for model, scenes in d.items():
        if not isinstance(scenes, dict):
            continue
        tp = fp = fn = 0
        ious = []
        for sc in scenes.values():
            if not isinstance(sc, dict):
                continue
            tp += sc.get("tp", 0)
            fp += sc.get("fp", 0)
            fn += sc.get("fn", 0)
            if sc.get("median_iou"):
                ious.append(sc["median_iou"])
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        b2 = 0.25
        f05 = ((1 + b2) * prec * rec / (b2 * prec + rec)
               if (b2 * prec + rec) else 0.0)
        rows.append({
            "модель": model, "TP": tp, "FP": fp, "FN": fn,
            "P": round(prec, 3), "R": round(rec, 3),
            "F1": round(f1, 3), "F0.5": round(f05, 3),
            "IoU_med": round(float(np.median(ious)), 3) if ious else None,
        })
    rows.sort(key=lambda r: -r["F0.5"])
    return rows


def speed_metrics() -> list[dict]:
    p = ROOT / "experiments/results/detector_benchmark.csv"
    d = _load(p)
    if not d:
        return []
    return [{"детектор": r.get("detector") or r.get("model"),
             "пристрій": r.get("device"),
             "P": r.get("precision"), "R": r.get("recall"),
             "FPS": r.get("fps")} for r in d]


def control_metrics() -> list[dict]:
    """Задача 3: стратегії керування в замкненому контурі."""
    p = ROOT / "experiments/results_hybrid/closed_loop.csv"
    d = _load(p)
    if not d:
        return []
    out = []
    for ctrl in sorted({r["controller"] for r in d}):
        sub = [r for r in d if r["controller"] == ctrl]
        ok = sum(1 for r in sub if r["success"] in ("True", "1", "true"))
        out.append({
            "стратегія": ctrl,
            "прогонів": len(sub),
            "успіх": f"{100*ok/len(sub):.0f}%",
            "смикання": round(float(np.mean(
                [float(r["jitter"]) for r in sub])), 4),
            "зусилля": round(float(np.mean(
                [float(r["effort"]) for r in sub])), 3),
        })
    return out


def latency_metrics() -> dict:
    """Новизна 1: компенсація латентності."""
    out = {}
    m = _load(ROOT / "experiments/results/latency/latency_measured.json")
    if m:
        out["затримка відео, мс"] = m.get("video_latency_ms")
        out["якість вимірювання"] = m.get("correlation")
        out["метод"] = m.get("method", "").replace("cross-correlation",
                                                    "крос-кореляція")
    e = _load(ROOT / "experiments/results/latency/latency_effect.csv")
    if e:
        for on, label in ((0, "без компенсації"), (1, "з компенсацією")):
            sub = [r for r in e if r["comp"] == str(on)]
            if not sub:
                continue
            out[f"смикання, {label}"] = round(float(np.mean(
                [float(r["d_yaw"]) for r in sub])), 2)
            out[f"|помилка|, {label}"] = round(float(np.median(
                [abs(float(r["err_true"])) for r in sub])), 3)
        a = out.get("смикання, без компенсації")
        b = out.get("смикання, з компенсацією")
        if a and b:
            out["зміна смикання"] = f"{(b-a)/a*100:+.0f}%"
    return out


def pass_metrics() -> dict:
    """Головна мета: проліт крізь обруч."""
    p = ROOT / "experiments/results/pass_trials.json"
    d = _load(p)
    if not d:
        return {"СТАТУС": "серія випробувань ще не проведена — "
                          "літати `python -m experiments.tello_fly "
                          "--mode approach --record` і зводити "
                          "результати вручну за оцінкою спостерігача"}
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="markdown для роботи")
    args = ap.parse_args()

    def table(title, rows, src):
        if not rows:
            print(f"\n## {title}\n\n  (даних немає: {src})")
            return
        print(f"\n## {title}")
        print(f"\nджерело: `{src}`, оновлено {_stamp(ROOT / src)}\n")
        keys = list(rows[0].keys())
        print("| " + " | ".join(keys) + " |")
        print("|" + "|".join(["---"] * len(keys)) + "|")
        for r in rows:
            print("| " + " | ".join(
                "—" if r.get(k) is None else str(r.get(k)) for k in keys) + " |")

    print("# Метрики магістерської роботи")
    print(f"\nзібрано {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    table("Задача 1. Детекція: точність на розмічених кадрах",
          detection_metrics(), "experiments/results/labeled_eval.json")
    table("Задача 1. Детекція: швидкість",
          speed_metrics(), "experiments/results/detector_benchmark.csv")
    table("Задача 3. Стратегії керування (замкнений контур)",
          control_metrics(), "experiments/results_hybrid/closed_loop.csv")

    lat = latency_metrics()
    print("\n## Новизна 1. Компенсація латентності (реальний дрон)\n")
    if lat:
        print("джерело: `experiments/results/latency/`, оновлено "
              f"{_stamp(ROOT / 'experiments/results/latency/latency_effect.csv')}\n")
        print("| показник | значення |")
        print("|---|---|")
        for k, v in lat.items():
            print(f"| {k} | {v} |")
    else:
        print("  (даних немає — запустити experiments/latency_on_drone.py)")

    pm = pass_metrics()
    print("\n## Головна мета. Проліт крізь обруч (реальний дрон)\n")
    print("| показник | значення |")
    print("|---|---|")
    for k, v in pm.items():
        print(f"| {k} | {v} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
