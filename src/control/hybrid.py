from __future__ import annotations

from .base import BaseController, ControlCommand
from .ibvs import IBVSController
from .pbvs import PBVSController


class HybridController(BaseController):
    """PBVS далеко -> IBVS близько, з незворотним перемиканням."""

    name = "hybrid"
    strategy = "hybrid"

    def __init__(
        self,
        pbvs: PBVSController,
        ibvs: IBVSController,
        switch_distance: float = 2.0,
        max_speed: float = 0.4,
        **kw,
    ) -> None:
        super().__init__(max_speed=max_speed, **kw)
        self.pbvs = pbvs
        self.ibvs = ibvs
        self.switch_distance = switch_distance
        self._on_ibvs = False
        self._switch_reason = ""

    def reset(self) -> None:
        super().reset()
        self.pbvs.reset()
        self.ibvs.reset()
        self._on_ibvs = False
        self._switch_reason = ""

    def _switch(self, reason: str) -> None:
        if not self._on_ibvs:
            self._on_ibvs = True
            self._switch_reason = reason

    def _annotate(self, cmd: ControlCommand, active: str) -> ControlCommand:
        cmd.diagnostics = dict(cmd.diagnostics)
        cmd.diagnostics["active"] = active
        cmd.diagnostics["switched"] = self._on_ibvs
        if self._switch_reason:
            cmd.diagnostics["switch_reason"] = self._switch_reason
        cmd.phase = f"{active.upper()}:{cmd.phase}"
        return cmd

    def _update_impl(self, detection, frame_shape, dt: float) -> ControlCommand:
        if self._on_ibvs:
            return self._annotate(self.ibvs.update(detection, frame_shape, dt), "ibvs")

        cmd = self.pbvs.update(detection, frame_shape, dt)

        if cmd.phase == "NO_POSE":
            self._switch("поза не відновлюється, метричне керування недоступне")
            return self._annotate(self.ibvs.update(detection, frame_shape, dt), "ibvs")

        distance = cmd.diagnostics.get("distance")
        if distance is not None and distance < self.switch_distance:
            self._switch(f"дистанція {distance:.2f}м < {self.switch_distance}м")
            return self._annotate(self.ibvs.update(detection, frame_shape, dt), "ibvs")

        return self._annotate(cmd, "pbvs")
