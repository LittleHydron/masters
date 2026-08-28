from __future__ import annotations

import numpy as np

from .base import BaseController, ControlCommand, PID
from ..pose.pnp_solver import PnPSolver, PoseEstimate


class PBVSController(BaseController):
    """Керування за метричною позою.

    Стратегія: спершу вийти на вісь обруча (approach_angle -> 0),
    тримаючи задану дистанцію, потім летіти вперед крізь центр.
    """

    name = "pbvs"
    strategy = "position_based"

    def __init__(
        self,
        solver: PnPSolver,
        max_speed: float = 0.4,
        kp_lateral: float = 0.8, ki_lateral: float = 0.03, kd_lateral: float = 0.15,
        kp_vertical: float = 0.8, ki_vertical: float = 0.03, kd_vertical: float = 0.15,
        kp_forward: float = 0.5, ki_forward: float = 0.0, kd_forward: float = 0.1,
        kp_yaw: float = 1.6, ki_yaw: float = 0.0, kd_yaw: float = 0.25,
        standoff_distance: float = 1.5,
        axis_tolerance_deg: float = 12.0,
        lateral_tolerance: float = 0.12,
        commit_distance: float = 1.2,
        blind_duration: float = 1.8,
    ) -> None:
        super().__init__(max_speed=max_speed)
        self.solver = solver
        self.pid_lateral = PID(kp_lateral, ki_lateral, kd_lateral)
        self.pid_vertical = PID(kp_vertical, ki_vertical, kd_vertical)
        self.pid_forward = PID(kp_forward, ki_forward, kd_forward)
        # Поворот, що утримує ціль у центрі кадру під час обльоту.
        self.pid_yaw = PID(kp_yaw, ki_yaw, kd_yaw)

        self.standoff_distance = standoff_distance
        self.axis_tolerance = np.deg2rad(axis_tolerance_deg)
        self.lateral_tolerance = lateral_tolerance
        self.commit_distance = commit_distance
        self.blind_duration = blind_duration

        self._blind_timer = 0.0
        self._orbit_side = 0.0
        self._lost_frames = 0
        self._committed = False
        self._prev_normal: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._orbit_side = 0.0
        for p in (self.pid_lateral, self.pid_vertical, self.pid_forward,
                  self.pid_yaw):
            p.reset()
        self._blind_timer = 0.0
        self._orbit_side = 0.0
        self._lost_frames = 0
        self._committed = False
        self._prev_normal = None

    def _disambiguate(self, pose: PoseEstimate) -> PoseEstimate:
        """Зняти двозначність кругової цілі часовою консистентністю.

        Розклад коніки дає дві дзеркальні нормалі. Обираємо ту, що
        ближча до нормалі з попереднього кадру: за час одного кадру
        реальна орієнтація змінюється мало, а хибна гілка стрибає.
        """
        if pose.alternative is None or self._prev_normal is None:
            if self._prev_normal is None:
                self._prev_normal = pose.normal.copy()
            return pose

        def consistency(p: PoseEstimate) -> float:
            return float(abs(np.dot(p.normal, self._prev_normal)))

        best = max([pose, pose.alternative], key=consistency)
        # Згладжуємо опорну нормаль, щоб один шумний кадр не перемкнув гілку.
        self._prev_normal = 0.7 * self._prev_normal + 0.3 * best.normal
        n = np.linalg.norm(self._prev_normal)
        if n > 1e-9:
            self._prev_normal /= n
        return best

    def _update_impl(self, detection, frame_shape, dt: float) -> ControlCommand:
        if detection is None:
            return self._handle_lost(dt)

        pose = self.solver.solve(detection)
        if pose is None:
            self._lost_frames += 1
            return ControlCommand(
                phase="NO_POSE", reason="детекція є, але поза не відновилась",
                diagnostics={"lost_frames": self._lost_frames},
            )

        pose = self._disambiguate(pose)
        self._lost_frames = 0

        dx, dy = pose.lateral_offset
        distance = pose.distance
        approach_angle = pose.approach_angle

        diagnostics = {
            "dx": dx, "dy": dy, "distance": distance,
            "approach_deg": float(np.degrees(approach_angle)),
            "ambiguous": pose.is_ambiguous,
        }

        if self._committed or distance < self.commit_distance:
            self._committed = True
            self._blind_timer = self.blind_duration
            return ControlCommand(
                lateral=float(np.clip(dx * 0.5, -self.max_speed, self.max_speed)),
                forward=self.max_speed,
                vertical=float(np.clip(-dy * 0.5, -self.max_speed, self.max_speed)),
                phase="COMMIT",
                reason=f"дистанція {distance:.2f}м < {self.commit_distance}м — проліт",
                diagnostics=diagnostics,
            )

        lateral = self.pid_lateral(dx, dt)
        vertical = -self.pid_vertical(dy, dt)

        on_axis = approach_angle < self.axis_tolerance
        centered = abs(dx) < self.lateral_tolerance and abs(dy) < self.lateral_tolerance

        n_to_cam = pose.normal if pose.normal[2] < 0 else -pose.normal
        ap = pose.position + n_to_cam * self.standoff_distance

        at_point = float(np.linalg.norm(ap)) < 0.5

        if on_axis and (centered or at_point):
            forward = self.pid_forward(distance - self.commit_distance, dt)
            return ControlCommand(
                lateral=lateral, forward=max(0.0, forward), vertical=vertical,
                phase="APPROACH",
                reason=f"на осі ({np.degrees(approach_angle):.0f}°), "
                       f"зближення до {self.commit_distance}м",
                diagnostics=diagnostics,
            )

        n_to_cam = pose.normal if pose.normal[2] < 0 else -pose.normal
        approach_point = pose.position + n_to_cam * self.standoff_distance

        ex, ey, ez = float(approach_point[0]), float(approach_point[1]), \
            float(approach_point[2])

        lateral = float(np.clip(self.pid_lateral(ex, dt),
                                -self.max_speed, self.max_speed))
        vertical = float(np.clip(-self.pid_vertical(ey, dt),
                                 -self.max_speed, self.max_speed))
        hold = float(np.clip(self.pid_forward(ez, dt),
                             -self.max_speed, self.max_speed))
        yaw = float(np.clip(self.pid_yaw(dx / max(0.5, distance), dt),
                            -1.0, 1.0))

        diagnostics["approach_err"] = float(np.linalg.norm([ex, ey, ez]))
        
        return ControlCommand(
            lateral=lateral,
            forward=hold,
            vertical=vertical,
            yaw=yaw,
            phase="ORBIT",
            reason=f"кут {np.degrees(approach_angle):.0f}° — наведення на "
                   f"точку підходу (похибка "
                   f"{diagnostics['approach_err']:.2f}м)",
            diagnostics=diagnostics,
        )

    def _handle_lost(self, dt: float) -> ControlCommand:
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
            reason=f"пошук цілі ({self._lost_frames} кадрів)",
            diagnostics={"lost_frames": self._lost_frames},
        )
