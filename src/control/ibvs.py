from __future__ import annotations

import numpy as np

from .base import (BaseController, ControlCommand, PID,
                   DeadzoneCompensator)
from ..detection.base import Detection


class IBVSController(BaseController):
    """Керування за помилками зображення.

    Фази:
      SEARCH  — ціль не видно, обертаємось на місці
      ALIGN   — зводимо центр цілі в центр кадру і робимо форму круглою
      APPROACH— форма достатньо кругла, рухаємось уперед, тримаючи центр
      BLIND   — ціль вийшла за межі кадру, доліт наосліп
    """

    name = "ibvs"
    strategy = "image_based"

    def __init__(
        self,
        max_speed: float = 0.4,
        
        kp_lateral: float = 0.7, ki_lateral: float = 0.0, kd_lateral: float = 0.12,
        kp_vertical: float = 0.7, ki_vertical: float = 0.02, kd_vertical: float = 0.12,
        kp_yaw: float = 0.5, ki_yaw: float = 0.0, kd_yaw: float = 0.08,
        
        roundness_target: float = 0.88,
        
        approach_fill: float = 0.55,
        blind_fill: float = 0.75,
        centering_tolerance: float = 0.08,
        vertical_bias: float = 0.0,
        min_lateral: float = 0.0,
        blind_duration: float = 1.8,
        near_throttle: float = 0.6,
        align_throttle: float = 0.25,
        **kw,
    ) -> None:
        super().__init__(max_speed=max_speed, **kw)
        self.pid_lateral = PID(kp_lateral, ki_lateral, kd_lateral)
        self.pid_vertical = PID(kp_vertical, ki_vertical, kd_vertical)
        self.pid_yaw = PID(kp_yaw, ki_yaw, kd_yaw)

        self.roundness_target = roundness_target
        self.approach_fill = approach_fill
        self.blind_fill = blind_fill
        self.centering_tolerance = centering_tolerance
        self.vertical_bias = vertical_bias
        self.min_lateral = min_lateral
        self._dz_lateral = DeadzoneCompensator(
            dead=min_lateral, eps=centering_tolerance * kp_lateral,
            limit=max_speed)
        self.blind_duration = blind_duration
        self.near_throttle = near_throttle
        self.align_throttle = align_throttle

        self._blind_timer = 0.0
        self._arc_dir = 0.0
        self._lost_frames = 0
        self._last_fill = 0.0

    def reset(self) -> None:
        super().reset()
        self._arc_dir = 0.0
        for p in (self.pid_lateral, self.pid_vertical, self.pid_yaw):
            p.reset()
        self._blind_timer = 0.0
        self._lost_frames = 0
        self._last_fill = 0.0
        self._dz_lateral.reset()

    @staticmethod
    def _shape_features(det: Detection) -> tuple[float, float]:
        """Повернути (roundness, skew_direction).

        roundness — 1.0 для ідеального кола, менше для сплюснутого.
        skew_direction — знак і величина сплюснутості по горизонталі:
        >0 означає, що еліпс вужчий по горизонталі, тобто дрон збоку.
        """
        if det.ellipse is not None:
            (_, _), (axis_a, axis_b), angle = det.ellipse
            major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
            if major <= 1e-6:
                return 0.0, 0.0
            roundness = minor / major

            theta = np.deg2rad(angle)
            if axis_a >= axis_b:
                major_dx = abs(np.sin(theta))
            else:
                major_dx = abs(np.cos(theta))
            skew = (1.0 - roundness) * (1.0 - 2.0 * major_dx)
            return float(roundness), float(skew)

        x1, y1, x2, y2 = det.bbox
        w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
        roundness = min(w, h) / max(w, h)
        skew = (1.0 - roundness) * (1.0 if w < h else -1.0)
        return float(roundness), float(skew)

    def _update_impl(self, detection, frame_shape, dt: float) -> ControlCommand:
        h, w = frame_shape[:2]

        if detection is None:
            return self._handle_lost(dt)

        self._lost_frames = 0
        cx, cy = detection.center

        err_x = (cx - w / 2.0) / (w / 2.0)
        err_y = (cy - h / 2.0) / (h / 2.0) - self.vertical_bias

        roundness, skew = self._shape_features(detection)
        
        x1, y1, x2, y2 = detection.bbox
        fill = max((x2 - x1) / w, (y2 - y1) / h)
        self._last_fill = fill

        diagnostics = {
            "err_x": err_x, "err_y": err_y,
            "roundness": roundness, "skew": skew, "fill": fill,
        }

        if fill >= self.blind_fill:
            self._blind_timer = self.blind_duration
            return ControlCommand(
                forward=self.max_speed, phase="BLIND",
                reason=f"ціль заповнює {fill:.0%} кадру, доліт наосліп",
                diagnostics=diagnostics,
            )

        vertical = -self.pid_vertical(err_y, dt)
        centered = abs(err_x) < self.centering_tolerance
        aligned = roundness >= self.roundness_target

        if aligned and centered:
            forward = self.max_speed * (1.0 if fill < self.approach_fill
                                        else self.near_throttle)
            return ControlCommand(
                lateral=self._lateral(err_x, dt),
                forward=forward,
                vertical=vertical,
                phase="APPROACH",
                reason=f"кругла {roundness:.2f}, по центру — вперед",
                diagnostics=diagnostics,
            )
        if self._arc_dir == 0.0:
            self._arc_dir = 1.0 if skew >= 0 else -1.0

        deficit = max(0.0, self.roundness_target - roundness)
        arc = float(np.clip(self._arc_dir * deficit * 2.0,
                            -self.max_speed * 0.7, self.max_speed * 0.7))
        lateral = float(np.clip(arc + self._lateral(err_x, dt),
                                -self.max_speed, self.max_speed))
        yaw = float(np.clip(self.pid_yaw(err_x, dt), -1.0, 1.0))

        forward = (0.0 if roundness < self.roundness_target * 0.9
                   else self.max_speed * self.align_throttle)

        return ControlCommand(
            lateral=lateral, forward=forward, vertical=vertical, yaw=yaw,
            phase="ALIGN",
            reason=f"круглість {roundness:.2f}<{self.roundness_target}, обліт, "
                   f"зсув ({err_x:+.2f},{err_y:+.2f})",
            diagnostics=diagnostics,
        )

    def _lateral(self, err_x: float, dt: float) -> float:
        """Бічна команда, дотягнута через мертву зону приводу.

        Дотяжку робить `DeadzoneCompensator` — афінним зсувом, а не
        підлогою. Підлога робила з каналу реле: будь-яка помилка, хоч
        0.002, хоч 0.14, давала однакові повні 8 (політ 01.10).
        """
        return self._dz_lateral(self.pid_lateral(err_x, dt))

    def _handle_lost(self, dt: float) -> ControlCommand:
        """Ціль зникла: або ми вже в сліпій фазі, або треба шукати."""
        self._lost_frames += 1

        if self._blind_timer > 0.0:
            self._blind_timer -= dt
            return ControlCommand(
                forward=self.max_speed, phase="BLIND",
                reason=f"проліт наосліп, лишилось {self._blind_timer:.1f}с",
                diagnostics={"blind_timer": self._blind_timer},
            )

        if self._lost_frames < 5:
            return ControlCommand(
                phase="HOLD", reason="ціль втрачено, утримання",
                diagnostics={"lost_frames": self._lost_frames},
            )

        return ControlCommand(
            yaw=0.25, phase="SEARCH",
            reason=f"пошук цілі ({self._lost_frames} кадрів без детекції)",
            diagnostics={"lost_frames": self._lost_frames},
        )
