from __future__ import annotations

from collections import defaultdict


class StickyTrackSelector:
    """Тримається одного треку, поки той присутній у кадрі.

    Перемикається лише коли обраний трек зник — тоді бере той, що
    має найдовшу історію.
    """

    def __init__(self) -> None:
        self.history: dict[int, int] = defaultdict(int)
        self.locked: int | None = None
        self.switches = 0

    def reset(self) -> None:
        self.history.clear()
        self.locked = None
        self.switches = 0

    def select(self, track_ids: list[int] | None,
               confidences: list[float]) -> int:
        """Повернути ІНДЕКС обраного бокса в списку детекцій.

        `track_ids` = None означає, що трекер не дав ID (перший кадр
        або режим predict) — тоді працюємо як звичайна детекція.
        """
        if not confidences:
            raise ValueError("порожній список детекцій")
        if not track_ids:
            return max(range(len(confidences)), key=lambda i: confidences[i])

        for t in track_ids:
            self.history[t] += 1

        if self.locked is not None and self.locked in track_ids:
            return track_ids.index(self.locked)

        best = max(range(len(track_ids)), key=lambda i: self.history[track_ids[i]])
        if self.locked is not None:
            self.switches += 1
        self.locked = track_ids[best]
        return best
