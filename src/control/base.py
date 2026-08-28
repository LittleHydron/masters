"""Спільний інтерфейс контролерів візуального сервоїнгу.

Дозволяє ганяти PBVS і IBVS на однакових кадрах і порівнювати
команди, які кожен видає в тій самій точці траєкторії.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class ControlCommand:
    """Команда керування в термінах rc-каналів Tello.

    Усі канали нормовані в [-1, 1] і масштабуються у [-100, 100]
    лише на межі з SDK. Це тримає контролери незалежними від дрона.
    """

    lateral: float = 0.0    # + управо
    forward: float = 0.0    # + вперед
    vertical: float = 0.0   # + вгору
    yaw: float = 0.0        # + за годинниковою

    # Пояснення рішення — потрібне саме для порівняння підходів.
    phase: str = "idle"
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)

    def clip(self, limit: float = 1.0) -> "ControlCommand":
        self.lateral = float(np.clip(self.lateral, -limit, limit))
        self.forward = float(np.clip(self.forward, -limit, limit))
        self.vertical = float(np.clip(self.vertical, -limit, limit))
        self.yaw = float(np.clip(self.yaw, -limit, limit))
        return self

    def to_rc(self) -> tuple[int, int, int, int]:
        """Перевести у формат send_rc_control (a, b, c, d) для Tello."""
        return (
            int(round(self.lateral * 100)),
            int(round(self.forward * 100)),
            int(round(self.vertical * 100)),
            int(round(self.yaw * 100)),
        )

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(
            [self.lateral, self.forward, self.vertical, self.yaw]
        ))

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("diagnostics", None)
        return d


class BaseController(ABC):
    """Базовий контролер.

    Контракт: `update` отримує детекцію (може бути None, якщо ціль
    загублено) і повертає команду. Контролер сам тримає свій стан.
    """

    name: str = "base"
    strategy: str = "base"

    def __init__(self, max_speed: float = 0.4) -> None:
        self.max_speed = max_speed
        self.history: list[ControlCommand] = []

    @abstractmethod
    def _update_impl(self, detection, frame_shape, dt: float) -> ControlCommand:
        ...

    def update(self, detection, frame_shape, dt: float = 1 / 30) -> ControlCommand:
        cmd = self._update_impl(detection, frame_shape, dt)
        cmd.clip(self.max_speed)
        self.history.append(cmd)
        return cmd

    def reset(self) -> None:
        self.history.clear()


class PID:
    """ПІД з захистом від накопичення інтегралу.

    Anti-windup обов'язковий: коли ціль надовго зникає з кадру,
    інтеграл інакше розганяється й дрон смикається при поверненні.
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 integral_limit: float = 1.0, output_limit: float = 1.0) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error: float | None = None

    def __call__(self, error: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-3

        self._integral = float(np.clip(
            self._integral + error * dt, -self.integral_limit, self.integral_limit
        ))

        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(out, -self.output_limit, self.output_limit))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
