"""Навчання YOLOv8n і YOLOv11n на датасеті обручів (задача 1).

Обидві моделі тренуються з ІДЕНТИЧНИМИ гіперпараметрами — інакше
порівняння архітектур перетворюється на порівняння налаштувань.

Параметри підібрані під GTX 1650 (4 ГБ VRAM).

Приклад:
    python -m experiments.train_detectors --data datasets/hoops/data.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ім'я збережених ваг = базове + суфікс з --tag: yolov8n_mixed.pt тощо.
# Фіксоване ім'я тут було б пасткою: прогін на іншому датасеті
# мовчки перезаписав би ваги попереднього.
MODELS = {
    "yolov8n": ("yolov8n.pt", "yolov8n"),
    "yolov11n": ("yolo11n.pt", "yolo11n"),
}


def train_one(key: str, data: str, epochs: int, imgsz: int, batch: int,
              device: str, workers: int, out_dir: Path, seed: int,
              tag: str, hsv_h: float = 0.015) -> dict:
    from ultralytics import YOLO

    pretrained, stem = MODELS[key]
    final_name = f"{stem}_{tag}.pt"
    print(f"\n{'=' * 64}\nНавчання {key}  (з {pretrained})\n{'=' * 64}")

    model = YOLO(pretrained)
    start = time.perf_counter()

    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        seed=seed,          # однаковий seed для обох моделей
        deterministic=True,
        project=str(out_dir / "runs"),
        name=key,
        exist_ok=True,
        patience=30,
        val=True,
        plots=True,
        # Аугментація ВІДТІНКУ. Типове 0.015 = зсув лише +-2.7 град зі
        # 180 — практично нуль, тоді як насиченість і яскравість
        # варіюються на 70% і 40%.
        #
        # На реальному відео (сцена Room, тепла жовта стіна) модель не
        # бачила червоний обруч ВЗАГАЛІ. Зсув відтінку фону на +60 у
        # тих самих кадрах підняв детекцію з 26% до 98% — тобто
        # відтінок і був вирішальним чинником, а модель його ніколи
        # не варіювала ні в даних, ні в аугментації.
        hsv_h=hsv_h,
        # AMP на GTX 1650 дає скромний приріст (Turing без повноцінних
        # tensor cores), але економить VRAM — при 4 ГБ це вирішальне.
        amp=True,
    )
    elapsed = time.perf_counter() - start

    metrics = model.val(data=data, imgsz=imgsz, device=device, verbose=False)

    # Ultralytics ігнорує відносний project і кладе прогін у
    # runs/detect/<project>/<name>. Питаємо шлях у самого тренера,
    # а не вгадуємо — інакше копіювання мовчки пропускається.
    best = None
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        cand = Path(save_dir) / "weights" / "best.pt"
        if cand.exists():
            best = cand
    if best is None:
        for cand in (out_dir / "runs" / key / "weights" / "best.pt",
                     Path("runs/detect") / out_dir / "runs" / key / "weights" / "best.pt"):
            if cand.exists():
                best = cand
                break
    if best is None:
        raise FileNotFoundError(
            f"Не знайдено best.pt для {key}. Навчання завершилось, але ваги "
            f"не скопійовано в models/{final_name}. Шукали в save_dir="
            f"{save_dir} і в {out_dir / 'runs' / key / 'weights'}."
        )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    shutil.copy(best, models_dir / final_name)
    print(f"Ваги -> {models_dir / final_name}  (з {best})")

    box = metrics.box
    result = {
        "model": key,
        "train_seconds": round(elapsed, 1),
        "epochs": epochs,
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "weights": str(models_dir / final_name),
    }
    print(f"\n{key}: mAP@50={result['mAP50']:.3f}  "
          f"mAP@50-95={result['mAP50_95']:.3f}  "
          f"P={result['precision']:.3f}  R={result['recall']:.3f}  "
          f"({elapsed / 60:.1f} хв)")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="шлях до data.yaml")
    ap.add_argument("--models", nargs="+", default=["yolov8n", "yolov11n"],
                    choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    # 16 має влізти в 4 ГБ на nano-моделях; при OOM зменшіть до 8.
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' для GPU, 'cpu' для CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hsv-h", type=float, default=0.015,
                    help="аугментація відтінку (0.015 типове ~ +-2.7 град; "
                         "0.3 ~ +-54 град — потрібно для стійкості до "
                         "теплих/холодних приміщень)")
    ap.add_argument("--tag", default="hoop",
                    help="суфікс імені ваг: models/yolov8n_<tag>.pt")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Не знайдено {data_path}\n"
            "Завантажте датасет із Roboflow у форматі YOLOv11 і розпакуйте "
            "в datasets/, потім вкажіть шлях до data.yaml."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [
        train_one(key, str(data_path), args.epochs, args.imgsz, args.batch,
                  args.device, args.workers, out_dir, args.seed, args.tag,
                  args.hsv_h)
        for key in args.models
    ]

    report = out_dir / "training_results.json"
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                      encoding="utf-8")

    print(f"\n{'=' * 64}\nПідсумок навчання\n{'=' * 64}")
    print(f"{'модель':<12} {'mAP@50':>8} {'mAP@50-95':>11} "
          f"{'precision':>10} {'recall':>8} {'час, хв':>9}")
    for r in results:
        print(f"{r['model']:<12} {r['mAP50']:>8.3f} {r['mAP50_95']:>11.3f} "
              f"{r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['train_seconds'] / 60:>9.1f}")
    print(f"\nЗбережено: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
