"""Оцінка затримки контуру ПІД ЧАС ПОЛЬОТУ, без магічних чисел.

Раніше `--latency` передавали рукою з окремого досліду. Це погано
двічі: число застаріває (Wi-Fi, завантаження ноутбука, температура
дрона) і його легко передати хибним — саме так і сталося 29.09,
коли замість 40 мс підставили 570 і компенсація погіршила керування.

Метод той самий, що в офлайн-вимірюванні, але рахується на льоту:
на ковзному вікні шукаємо часовий зсув, за якого швидкість курсу
(телеметрія, майже без затримки) найкраще пояснює швидкість цілі
в кадрі (проходить через відеотракт).

Курс і піксель рухаються в ПРОТИЛЕЖНІ боки — поворот праворуч жене
ціль ліворуч, — тому шукаємо найсильнішу НЕГАТИВНУ кореляцію.

Оцінка оновлюється лише тоді, коли в вікні був реальний рух: на
нерухомій цілі кореляція шумова й дала б випадкове число. Доки
надійної оцінки немає, віддається `fallback`.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class LatencyEstimator:
    """Ковзна оцінка затримки відеотракту, секунди.

    Приклад:
        est = LatencyEstimator()
        est.update(t, yaw_deg, cx_px)
        lat = est.latency          # завжди безпечне число
    """

    def __init__(self, *, window: float = 4.0, step: float = 0.02,
                 max_lag: float = 0.6, fallback: float = 0.05,
                 min_corr: float = 0.25, min_motion: float = 8.0,
                 smoothing: float = 0.7) -> None:
        self.window = window
        self.step = step
        self.max_lag = max_lag
        self.fallback = fallback
        self.min_corr = min_corr
        self.min_motion = min_motion
        self.smoothing = smoothing
        self._buf: deque[tuple[float, float, float]] = deque()
        self._lat: float | None = None
        self._corr = 0.0
        self._n_updates = 0


    def update(self, t: float, yaw_deg: float, cx: float) -> None:
        self._buf.append((t, yaw_deg, cx))
        while self._buf and t - self._buf[0][0] > self.window:
            self._buf.popleft()
        if len(self._buf) >= 40 and len(self._buf) % 10 == 0:
            self._recompute()

    def _recompute(self) -> None:
        a = np.array(self._buf, dtype=float)
        t = a[:, 0] - a[0, 0]
        if t[-1] < self.window * 0.5:
            return

        grid = np.arange(0, t[-1], self.step)
        if len(grid) < 20:
            return
        yaw = np.interp(grid, t, np.unwrap(np.deg2rad(a[:, 1])))
        cx = np.interp(grid, t, a[:, 2])

        dy = np.diff(yaw)
        dc = np.diff(cx)
        if np.std(dc) / self.step < self.min_motion:
            return
        if np.std(dy) < 1e-6 or np.std(dc) < 1e-6:
            return
        dy = (dy - dy.mean()) / dy.std()
        dc = (dc - dc.mean()) / dc.std()

        n_lag = int(self.max_lag / self.step)
        corrs = []
        for L in range(n_lag):
            m = len(dy) - L
            if m < 15:
                break
            corrs.append(float(np.dot(dy[:m], dc[L:L + m])) / m)
        if len(corrs) < 3:
            return
        best_lag = int(np.argmin(corrs))
        best_corr = corrs[best_lag]

        if -best_corr < self.min_corr:
            return

        lat = best_lag * self.step
        if 0 < best_lag < len(corrs) - 1:
            y0, y1, y2 = corrs[best_lag - 1], corrs[best_lag], corrs[best_lag + 1]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-12:
                lat += 0.5 * (y0 - y2) / denom * self.step
        lat = float(np.clip(lat, 0.0, self.max_lag))
        self._lat = lat if self._lat is None else (
            self.smoothing * self._lat + (1 - self.smoothing) * lat)
        self._corr = best_corr
        self._n_updates += 1


    @property
    def latency(self) -> float:
        """Поточна оцінка, або fallback доки її немає."""
        return self.fallback if self._lat is None else float(self._lat)

    @property
    def ready(self) -> bool:
        return self._lat is not None

    @property
    def quality(self) -> float:
        """|кореляція| останньої вдалої оцінки: 0 = немає, 1 = ідеал."""
        return abs(self._corr)

    @property
    def updates(self) -> int:
        return self._n_updates

    def reset(self) -> None:
        self._buf.clear()
        self._lat = None
        self._corr = 0.0
        self._n_updates = 0
