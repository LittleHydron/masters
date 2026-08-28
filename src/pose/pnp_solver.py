from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..detection.base import Detection, TargetShape


@dataclass
class PoseEstimate:
    """Поза цілі у системі координат камери. Відстані в метрах."""

    position: np.ndarray        # (3,) вектор на центр цілі
    normal: np.ndarray          # (3,) одинична нормаль площини цілі
    rotation: np.ndarray | None = None  # (3,3), лише для RECTANGLE
    reprojection_error: float = 0.0
    is_ambiguous: bool = False
    alternative: "PoseEstimate | None" = None
    method: str = ""

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.position))

    @property
    def lateral_offset(self) -> tuple[float, float]:
        return float(self.position[0]), float(self.position[1])

    @property
    def yaw_to_target(self) -> float:
        return float(np.arctan2(self.position[0], self.position[2]))

    @property
    def approach_angle(self) -> float:
        los = self.position / max(1e-9, np.linalg.norm(self.position))
        cos_a = float(np.clip(abs(np.dot(self.normal, los)), -1.0, 1.0))
        return float(np.arccos(cos_a))


class PnPSolver:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
        target_diameter: float = 0.60,
        target_size: tuple[float, float] = (0.60, 0.60),
    ) -> None:
        self.K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist = (
            np.zeros((5, 1), dtype=np.float64)
            if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        )
        self.target_diameter = float(target_diameter)
        self.target_size = target_size

    def solve(self, detection: Detection) -> PoseEstimate | None:
        if detection.shape is TargetShape.RECTANGLE:
            return self._solve_rectangle(detection)
        return self._solve_circle(detection)

    def _solve_rectangle(self, detection: Detection) -> PoseEstimate | None:
        if detection.corners is None:
            return None

        w, h = self.target_size
        
        object_points = np.array(
            [
                [-w / 2, -h / 2, 0.0],
                [+w / 2, -h / 2, 0.0],
                [+w / 2, +h / 2, 0.0],
                [-w / 2, +h / 2, 0.0],
            ],
            dtype=np.float64,
        )
        image_points = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.K, self.dist)
        error = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - image_points, axis=1)))

        return PoseEstimate(
            position=tvec.ravel(),
            normal=R[:, 2],
            rotation=R,
            reprojection_error=error,
            is_ambiguous=False,
            method="solvepnp_ippe_square",
        )

    def _solve_circle(self, detection: Detection) -> PoseEstimate | None:
        if detection.ellipse is None:
            return None

        Q = self._ellipse_to_conic(detection.ellipse)
        
        Q_n = self.K.T @ Q @ self.K
        Q_n = (Q_n + Q_n.T) / 2.0
        norm = np.linalg.norm(Q_n)
        if norm < 1e-15:
            return None
        Q_n /= norm

        eigenvalues, eigenvectors = np.linalg.eigh(Q_n)

        signs = np.sign(eigenvalues)
        odd = [i for i in range(3)
               if signs[i] != signs[(i + 1) % 3] and signs[i] != signs[(i + 2) % 3]]
        if len(odd) != 1:
            return None
        i3 = odd[0]
        pair = [i for i in range(3) if i != i3]
        
        if abs(eigenvalues[pair[0]] - eigenvalues[i3]) >= abs(eigenvalues[pair[1]] - eigenvalues[i3]):
            i1, i2 = pair[0], pair[1]
        else:
            i1, i2 = pair[1], pair[0]

        l1, l2, l3 = eigenvalues[i1], eigenvalues[i2], eigenvalues[i3]
        e1, e3 = eigenvectors[:, i1], eigenvectors[:, i3]

        if abs(l1 - l3) < 1e-15 or abs(l2) < 1e-15:
            return None

        g = np.sqrt(max(0.0, (l1 - l2) / (l1 - l3)))
        k = np.sqrt(max(0.0, (l2 - l3) / (l1 - l3)))

        radius = self.target_diameter / 2.0
        z0 = radius * np.sqrt(max(1e-15, abs(l2) / abs(l3)))

        candidates: list[PoseEstimate] = []
        for sign in (+1.0, -1.0):
            direction = e3
            center = z0 * direction / np.linalg.norm(direction)

            normal = k * e3 + sign * g * e1
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-9:
                continue
            normal = normal / n_norm

            if center[2] < 0:
                center = -center
                normal = -normal
            if np.dot(normal, center) > 0:
                normal = -normal

            candidates.append(
                PoseEstimate(
                    position=center,
                    normal=normal,
                    is_ambiguous=True,
                    method="conic_decomposition",
                )
            )

        if not candidates:
            return None

        primary = candidates[0]
        if len(candidates) > 1:
            primary.alternative = candidates[1]
        return primary

    @staticmethod
    def _ellipse_to_conic(ellipse) -> np.ndarray:
        """Перевести (center, axes, angle) з OpenCV у матрицю коніки 3x3.

        Коніка задає точки x, для яких x^T Q x = 0, у однорідних
        піксельних координатах.
        """
        (cx, cy), (major, minor), angle_deg = ellipse
        a = major / 2.0
        b = minor / 2.0
        theta = np.deg2rad(angle_deg)

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        A = (cos_t ** 2) / (a ** 2) + (sin_t ** 2) / (b ** 2)
        B = 2 * cos_t * sin_t * (1.0 / (a ** 2) - 1.0 / (b ** 2))
        C = (sin_t ** 2) / (a ** 2) + (cos_t ** 2) / (b ** 2)
        D = -2 * A * cx - B * cy
        E = -B * cx - 2 * C * cy
        F = A * cx ** 2 + B * cx * cy + C * cy ** 2 - 1.0

        return np.array(
            [
                [A, B / 2.0, D / 2.0],
                [B / 2.0, C, E / 2.0],
                [D / 2.0, E / 2.0, F],
            ],
            dtype=np.float64,
        )
