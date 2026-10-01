"""Компенсація латентності: керування за прогнозованим, а не застарілим станом.

Заявлена новизна роботи. Задача: кадр, який контролер бачить зараз,
описує стан дрона, який був `latency` секунд тому — кодування H.264,
Wi-Fi туди-назад, декодування, інференс. ПІД реагує на минуле, і на
швидкому зближенні це дає перерегулювання: дрон гальмує тоді, коли
вже проскочив.

Рішення — екстраполювати спостереження вперед на час затримки, перш
ніж віддати його контролеру.

Два рівні, свідомо розділені:

* `LinearPredictor` — лінійна екстраполяція за швидкістю, порахованою
  з двох останніх спостережень. Просто, без параметрів, працює
  одразу. Базовий варіант для порівняння.
* `KalmanPredictor` — фільтр Калмана з моделлю постійної швидкості.
  На відміну від лінійного, зважує довіру до вимірів проти моделі,
  тому переживає шум детекції й пропуски кадрів.

Обидва реалізують той самий інтерфейс, тому порівнюються на однаковому
стенді: `closed_loop_sim.py --latency 0.20` з компенсацією і без.

Стан — це вектор спостереження (напр. [cx, cy, distance]); модуль не
знає, що саме в ньому, і однаково годиться для IBVS (пікселі) та
PBVS (метри).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class BasePredictor:
    """Спільний інтерфейс: подати спостереження -> дістати прогноз."""

    def update(self, z: np.ndarray, t: float) -> None:
        raise NotImplementedError

    def predict(self, horizon: float) -> np.ndarray | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


@dataclass
class LinearPredictor(BasePredictor):
    """Лінійна екстраполяція: z(t+h) = z(t) + v*h.

    Швидкість — скінченна різниця двох останніх спостережень зі
    згладжуванням, бо сира різниця на шумі детекції стрибає.
    """

    smoothing: float = 0.6
    max_horizon: float = 0.35
    max_shift: float | None = 60.0
    _z: np.ndarray | None = None
    _t: float | None = None
    _v: np.ndarray | None = None

    def update(self, z: np.ndarray, t: float) -> None:
        z = np.asarray(z, dtype=float)
        if self._z is not None and self._t is not None:
            dt = t - self._t
            if dt > 1e-4:
                v = (z - self._z) / dt
                self._v = v if self._v is None else (
                    self.smoothing * self._v + (1 - self.smoothing) * v)
        self._z, self._t = z, t

    def predict(self, horizon: float) -> np.ndarray | None:
        if self._z is None:
            return None
        if self._v is None:
            return self._z.copy()
        h = float(np.clip(horizon, 0.0, self.max_horizon))
        d = self._v * h
        if self.max_shift is not None:
            n = float(np.linalg.norm(d))
            if n > self.max_shift and n > 0:
                d = d / n * self.max_shift
        return self._z + d

    def reset(self) -> None:
        self._z = self._t = self._v = None


class KalmanPredictor(BasePredictor):
    """Фільтр Калмана, модель постійної швидкості.

    Стан: [позиція, швидкість] для кожної компоненти спостереження.
    Переваги над лінійним: зважує вимір проти моделі за їхніми
    коваріаціями, тому шумні або пропущені детекції не викидають
    прогноз. Саме це потрібно, бо детектор реально губить ціль
    (на реальних відео — окремі кадри, див. masters-detection-problems).

    Параметри задаються в одиницях спостереження:
      q - наскільки довіряємо моделі (більше = модель гірша)
      r - наскільки шумний вимір (більше = вимір гірший)

    Дефолт q=500 підібрано для спостережень у ПІКСЕЛЯХ (кадр 960x720,
    ціль рухається сотнями пікселів за секунду). З q=1 фільтр не вірив
    у рух узагалі: оцінка швидкості 38 замість 120 px/с, і прогноз
    виходив гіршим за відсутність компенсації. Для спостережень у
    МЕТРАХ масштаб інший — там q близько 1-10.
    """

    def __init__(self, dim: int, *, q: float = 500.0, r: float = 4.0,
                 max_horizon: float = 0.35,
                 max_shift: float | None = 60.0) -> None:
        self.dim = dim
        self.q, self.r = q, r
        self.max_horizon = max_horizon
        self.max_shift = max_shift
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._t: float | None = None

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(2 * self.dim)
        for i in range(self.dim):
            F[i, self.dim + i] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        """Шум процесу для моделі постійної швидкості (білий по прискоренню)."""
        Q = np.zeros((2 * self.dim, 2 * self.dim))
        for i in range(self.dim):
            Q[i, i] = self.q * dt ** 3 / 3.0
            Q[i, self.dim + i] = self.q * dt ** 2 / 2.0
            Q[self.dim + i, i] = self.q * dt ** 2 / 2.0
            Q[self.dim + i, self.dim + i] = self.q * dt
        return Q

    def update(self, z: np.ndarray, t: float) -> None:
        z = np.asarray(z, dtype=float)
        if self._x is None:
            self._x = np.concatenate([z, np.zeros(self.dim)])
            self._P = np.eye(2 * self.dim) * 10.0
            self._t = t
            return

        dt = max(1e-4, t - self._t)
        self._t = t
        F = self._F(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q(dt)

        H = np.zeros((self.dim, 2 * self.dim))
        H[:, :self.dim] = np.eye(self.dim)
        R = np.eye(self.dim) * self.r
        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(2 * self.dim) - K @ H) @ self._P

    def predict(self, horizon: float) -> np.ndarray | None:
        if self._x is None:
            return None
        h = float(np.clip(horizon, 0.0, self.max_horizon))
        pred = (self._F(h) @ self._x)[:self.dim]
        return self._limit(pred, self._x[:self.dim])

    def _limit(self, pred: np.ndarray, now: np.ndarray) -> np.ndarray:
        """Обмежити зсув прогнозу.

        На реальних даних (політ 29.09) фільтр видавав зсув до 1317 px
        при ширині кадру 960: сплеск швидкості від шуму детекції
        розганяв оцінку, і ПІД отримував помилку поза межами кадру.
        Результат — смикання +31% замість покращення.

        Обмеження робить компенсацію безпечною: вона може лише
        підправити спостереження, а не вигадати нове.
        """
        if self.max_shift is None:
            return pred
        d = pred - now
        n = float(np.linalg.norm(d))
        if n > self.max_shift and n > 0:
            d = d / n * self.max_shift
        return now + d

    def reset(self) -> None:
        self._x = self._P = self._t = None


def make_predictor(kind: str, dim: int, **kw) -> BasePredictor | None:
    """Фабрика за назвою: none / linear / kalman."""
    kind = (kind or "none").lower()
    if kind in ("none", "off", ""):
        return None
    if kind == "linear":
        return LinearPredictor(**kw)
    if kind == "kalman":
        return KalmanPredictor(dim, **kw)
    raise ValueError(f"невідомий компенсатор: {kind}")
