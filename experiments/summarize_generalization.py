"""Звести крос-датасетні результати в таблицю: чи допомогла синтетика.

Ключове питання роботи: висока точність на рідному тесті — це якість
детекції чи витік даних? Відповідь дає падіння F1 на чужих датасетах.
Скрипт рахує це падіння для кожного варіанта навчання.

Приклад:
    python -m experiments.summarize_generalization \
        --json experiments/results/cross_dataset_mixed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Три категорії перевірок — НЕ змішувати в одну середню.
#  HOLDOUT — інший обруч тієї ж природи (те, що потрібно роботі)
#  CROSS   — інший ТИП об'єкта: кільця FRC згори (ширша перевірка)
HOLDOUT = ("synth_holdout",)
CROSS = ("notedetection", "note_detection")


def category(dataset: str) -> str:
    if dataset in HOLDOUT:
        return "holdout"
    if dataset in CROSS:
        return "cross"
    return "native"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", nargs="+", required=True,
                    help="один або більше cross_dataset_*.json")
    ap.add_argument("--out", default="experiments/results/generalization.md")
    args = ap.parse_args()

    rows: list[dict] = []
    for p in args.json:
        rows.extend(json.loads(Path(p).read_text(encoding="utf-8")))

    # Групуємо за моделлю: рідний F1 vs середній чужий F1.
    by_model: dict[str, dict] = {}
    for r in rows:
        m = by_model.setdefault(r["model"],
                                {"native": [], "holdout": [], "cross": []})
        m[category(r["dataset"])].append(r["f1"])

    avg = lambda v: sum(v) / len(v) if v else None
    lines = [
        "| варіант навчання | F1 рідний | F1 інший обруч | падіння | "
        "F1 інший тип | падіння |",
        "|---|---|---|---|---|---|",
    ]
    table = []
    for name, d in by_model.items():
        nat = avg(d["native"])
        if nat is None:
            continue
        hold, cross = avg(d["holdout"]), avg(d["cross"])
        dh = (nat - hold) / nat * 100 if hold is not None and nat else None
        dc = (nat - cross) / nat * 100 if cross is not None and nat else None
        table.append((name, nat, hold, dh, cross, dc))

    # Сортуємо за падінням на holdout — головна цифра роботи.
    table.sort(key=lambda t: (t[3] is None, t[3] if t[3] is not None else 0))
    fmt = lambda v, suf="": f"{v:.3f}{suf}" if v is not None else "—"
    for name, nat, hold, dh, cross, dc in table:
        lines.append(
            f"| {name} | {nat:.3f} | {fmt(hold)} | "
            f"{f'{dh:.1f}%' if dh is not None else '—'} | "
            f"{fmt(cross)} | {f'{dc:.1f}%' if dc is not None else '—'} |")

    text = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")

    print(f"{'варіант':<18}{'рідний':>9}{'iнш.обруч':>11}{'падiння':>9}"
          f"{'iнш.тип':>10}{'падiння':>9}")
    print("=" * 66)
    for name, nat, hold, dh, cross, dc in table:
        h = f"{hold:.3f}" if hold is not None else "—"
        c = f"{cross:.3f}" if cross is not None else "—"
        sh = f"{dh:.1f}%" if dh is not None else "—"
        sc = f"{dc:.1f}%" if dc is not None else "—"
        print(f"{name:<18}{nat:>9.3f}{h:>11}{sh:>9}{c:>10}{sc:>9}")
    print(f"\nЗбережено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
