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

    lateral: float = 0.0
    forward: float = 0.0
    vertical: float = 0.0
    yaw: float = 0.0

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

    def __init__(self, max_speed: float = 0.4, *,
                 predictor=None, latency: float = 0.0) -> None:
        self.max_speed = max_speed
        self.history: list[ControlCommand] = []
        self.predictor = predictor
        self.latency = float(latency)
        self._pred_t = 0.0

    @abstractmethod
    def _update_impl(self, detection, frame_shape, dt: float) -> ControlCommand:
        ...

    def update(self, detection, frame_shape, dt: float = 1 / 30) -> ControlCommand:
        if self.predictor is not None and detection is not None:
            detection = self._compensate(detection, dt)
        cmd = self._update_impl(detection, frame_shape, dt)
        cmd.clip(self.max_speed)
        self.history.append(cmd)
        return cmd

    def _compensate(self, detection, dt: float):
        """Зсунути детекцію вперед на час затримки контуру.

        Кадр описує стан, який був `latency` секунд тому. Без зсуву
        ПІД реагує на минуле й перерегульовує. Екстраполюємо ЦЕНТР
        рамки, зберігаючи її розмір: розмір міняється повільніше за
        положення, а його прогноз вносив би похибку в оцінку пози.

        Підміняємо копію, а не оригінал: сира детекція потрібна для
        логів і порівняння "що бачили / куди цілимось".
        """
        import copy

        cx, cy = detection.center
        self._pred_t += dt
        self.predictor.update(np.array([cx, cy]), self._pred_t)
        pred = self.predictor.predict(self.latency)
        if pred is None:
            return detection

        dx, dy = float(pred[0] - cx), float(pred[1] - cy)
        x1, y1, x2, y2 = detection.bbox
        out = copy.copy(detection)
        out.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        if detection.ellipse is not None:
            (ex, ey), axes, ang = detection.ellipse
            out.ellipse = ((ex + dx, ey + dy), axes, ang)
        out.meta = dict(detection.meta)
        out.meta["compensated"] = (round(dx, 1), round(dy, 1))
        return out

    def reset(self) -> None:
        self.history.clear()
        if self.predictor is not None:
            self.predictor.reset()
        self._pred_t = 0.0


class DeadzoneCompensator:
    """Дотяжка команди через мертву зону приводу.

    У Tello бічний канал нижче ~7 зі 100 не зрушує дрона. Наївне
    рішення — підняти будь-яку малу команду до порога — перетворює
    канал у РЕЛЕ: політ 01.10 дав lat = -8 у 355 кадрах з 361 у фазі
    ALIGN і клацання +8/-8 (65 проти 61) у APPROACH. Пропорційності
    не лишилось, дрона розкачало поперек курсу, і він зачепив косяк.

    Тут замість порога — АФІННИЙ ЗСУВ: нуль лишається нулем, а все
    поза зоною нечутливості `eps` відображається з `dead` і далі
    росте пропорційно до того самого максимуму:

        u = 0                                        , |u| <= eps
        u = sign(u) * (dead + (|u|-eps) * k)         , інакше
        k = (limit - dead) / (limit - eps)

    `eps` потрібен, щоб шум детектора не давав ривків на `dead`, а
    гістерезис (`_sign`) не дає каналу клацати знаком біля центру:
    щоб змінити напрямок, помилка має перейти нуль з запасом.
    """

    def __init__(self, dead: float = 0.0, eps: float = 0.02,
                 limit: float = 1.0, hysteresis: float = 0.5) -> None:
        self.dead = max(0.0, float(dead))
        self.eps = max(0.0, float(eps))
        self.limit = float(limit)
        self.hysteresis = float(hysteresis)
        self._sign = 0.0

    def reset(self) -> None:
        self._sign = 0.0

    def __call__(self, u: float) -> float:
        u = float(u)
        if self.dead <= 0.0:
            return u

        mag = abs(u)
        if mag <= self.eps:
            self._sign = 0.0
            return 0.0

        sign = 1.0 if u > 0.0 else -1.0
        if self._sign != 0.0 and sign != self._sign:
            if mag < self.eps * (1.0 + self.hysteresis):
                return 0.0
        self._sign = sign

        span = max(1e-6, self.limit - self.eps)
        k = (self.limit - self.dead) / span
        out = self.dead + (mag - self.eps) * k
        return float(sign * min(self.limit, out))


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
