"""Модель руху дрона: інерція, затримка керування, обмеження.

Замінює кінематику «команда = миттєве переміщення», яка робить
порівняння стратегій нечесним: контролер, що смикає команди, у
кінематиці не платить за це нічого, а на реальному дроні платить
перерегулюванням.

Три ефекти, які тут з'являються і яких не було раніше:

1. **Інерція.** Команда задає ЦІЛЬОВУ швидкість; фактична підтягується
   до неї з обмеженим прискоренням. Різка зміна команди не дає різкої
   зміни руху.
2. **Затримка.** Команда діє не одразу, а через `latency` секунд —
   рівно та проблема, заради якої в роботі заявлено компенсацію
   латентності. Реалізовано чергою: команда стає в хвіст, виймається
   з голови через потрібну кількість кроків.
3. **Обмеження.** Максимальна швидкість і прискорення — з паспорта
   Tello, а не з голови.

Модель навмисно проста: без аеродинаміки, ground effect і downwash.
Мета — чесно порівняти СТРАТЕГІЇ між собою, а не відтворити політ
Tello. Для повної аеродинаміки потрібен PyFlyt/PyBullet (див.
дослідження в STATUS.md).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class DroneParams:
    """Параметри руху. Типові значення — DJI Robomaster TT (Tello).

    max_speed: у режимі rc_control Tello обмежений приблизно 1 м/с
        на канал; беремо трохи вище, бо дозволяємо діагональ.
    max_accel: оцінка знизу для нано-квадрокоптера. Занижувати
        безпечніше: так стенд не лестить різким контролерам.
    latency: сумарна затримка контуру — кодування відео, Wi-Fi туди
        й назад, обробка. Для Tello по Wi-Fi реально 0.15-0.30 с.
    """

    max_speed: float = 1.2          # м/с на канал
    max_yaw_rate: float = 1.2       # рад/с — поворот навколо вертикалі
    max_accel: float = 2.5          # м/с² — підтягування до цілі
    latency: float = 0.20           # с — затримка команди
    drag: float = 0.35              # 1/с — гальмування при нульовій команді
    speed_noise: float = 0.02       # м/с — неідеальність приводу


class DroneModel:
    """Стан дрона й інтегрування команд у рух.

    Система координат — камерна: x управо, y вниз, z вперед.
    Команда `ControlCommand` нормована в [-1, 1] на канал.

    Дрон має КУРС (`yaw`): без нього обліт цілі неможливий у принципі.
    Щоб обійти обруч і не втратити його з кадру, камера мусить
    повертатися; інакше будь-який бічний рух виводить ціль за межі
    поля зору, і фази вирівнювання не завершуються ніколи.
    """

    def __init__(self, position: np.ndarray, params: DroneParams | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.p = np.asarray(position, dtype=float).copy()
        self.v = np.zeros(3)
        # Курс у радіанах: 0 = камера дивиться вздовж +z світу.
        self.yaw = 0.0
        self.params = params or DroneParams()
        self.rng = rng or np.random.default_rng(0)
        self._queue: deque[np.ndarray] = deque()
        self._delay_steps: int | None = None

    def reset(self, position: np.ndarray) -> None:
        self.p = np.asarray(position, dtype=float).copy()
        self.v = np.zeros(3)
        self.yaw = 0.0
        self._queue.clear()
        self._delay_steps = None

    def step(self, cmd, dt: float) -> None:
        """Застосувати команду з урахуванням затримки та інерції."""
        pr = self.params

        # Курс змінюється одразу (поворот швидший за переміщення).
        self.yaw += float(cmd.yaw) * pr.max_yaw_rate * dt

        # Команда -> цільова швидкість у ВЛАСНІЙ системі дрона,
        # далі повертаємо її у світову за поточним курсом.
        local = np.array([cmd.lateral, -cmd.vertical, cmd.forward]) * pr.max_speed
        c, sn = np.cos(self.yaw), np.sin(self.yaw)
        target = np.array([c * local[0] + sn * local[2],
                           local[1],
                           -sn * local[0] + c * local[2]])

        # Затримка: скільки кроків тримати команду в черзі.
        if self._delay_steps is None:
            self._delay_steps = max(0, int(round(pr.latency / dt)))
        self._queue.append(target)
        if len(self._queue) > self._delay_steps:
            applied = self._queue.popleft()
        else:
            # Контур ще не «прогрітий» — дрон стоїть, команди в дорозі.
            applied = np.zeros(3)

        # Інерція: підтягуємо фактичну швидкість до цільової з
        # обмеженим прискоренням.
        dv = applied - self.v
        step_limit = pr.max_accel * dt
        n = float(np.linalg.norm(dv))
        if n > step_limit and n > 0:
            dv = dv / n * step_limit
        self.v = self.v + dv

        # Гальмування й неідеальність приводу.
        self.v *= max(0.0, 1.0 - pr.drag * dt)
        if pr.speed_noise > 0:
            self.v = self.v + self.rng.normal(0, pr.speed_noise, 3) * dt

        self.p = self.p + self.v * dt

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))

    def to_camera(self, point_world: np.ndarray) -> np.ndarray:
        """Перевести точку зі світових координат у систему камери."""
        rel = np.asarray(point_world, dtype=float) - self.p
        c, sn = np.cos(-self.yaw), np.sin(-self.yaw)
        return np.array([c * rel[0] + sn * rel[2],
                         rel[1],
                         -sn * rel[0] + c * rel[2]])

    def rotate_to_camera(self, vec_world: np.ndarray) -> np.ndarray:
        """Повернути вектор (без зсуву) у систему камери."""
        v = np.asarray(vec_world, dtype=float)
        c, sn = np.cos(-self.yaw), np.sin(-self.yaw)
        return np.array([c * v[0] + sn * v[2], v[1], -sn * v[0] + c * v[2]])
