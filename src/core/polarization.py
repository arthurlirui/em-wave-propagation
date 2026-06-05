"""
偏振追踪 — Stokes 矢量 & Mueller 矩阵

在全链路中追踪电磁波的偏振态变化。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class StokesVector:
    """Stokes 矢量 S = [I, Q, U, V]^T

    I: 总强度
    Q: 水平/垂直线偏振
    U: ±45° 线偏振
    V: 左/右旋圆偏振
    """

    def __init__(self, i: float = 1.0, q: float = 0.0, u: float = 0.0, v: float = 0.0):
        self.v = np.array([i, q, u, v], dtype=np.float64)

    @property
    def intensity(self) -> float:
        return self.v[0]

    @property
    def dop(self) -> float:
        """偏振度 Degree of Polarization (0-1)"""
        if self.v[0] < 1e-15:
            return 0.0
        return np.sqrt(self.v[1]**2 + self.v[2]**2 + self.v[3]**2) / self.v[0]

    @property
    def lop(self) -> float:
        """线偏振度"""
        if self.v[0] < 1e-15:
            return 0.0
        return np.sqrt(self.v[1]**2 + self.v[2]**2) / self.v[0]

    @property
    def circular_ratio(self) -> float:
        """圆偏振比 V/I"""
        return self.v[3] / max(self.v[0], 1e-15)

    def apply_mueller(self, m: np.ndarray) -> StokesVector:
        """应用 4×4 Mueller 矩阵"""
        return StokesVector(*m @ self.v)

    def __repr__(self) -> str:
        return f"Stokes(I={self.v[0]:.4f}, Q={self.v[1]:.4f}, U={self.v[2]:.4f}, V={self.v[3]:.4f})"


class MuellerMatrix:
    """Mueller 矩阵工厂方法"""

    @staticmethod
    def fresnel_reflection(r_te: float, r_tm: float) -> np.ndarray:
        """Fresnel 反射的 Mueller 矩阵（假设坐标系对齐到 s/p 方向）

        Args:
            r_te: TE 反射系数（幅度）
            r_tm: TM 反射系数（幅度）
        """
        R_s = abs(r_te)**2
        R_p = abs(r_tm)**2
        return np.array([
            [ (R_s + R_p) / 2,  (R_s - R_p) / 2,  0,  0],
            [ (R_s - R_p) / 2,  (R_s + R_p) / 2,  0,  0],
            [0, 0, np.sqrt(R_s * R_p), 0],
            [0, 0, 0, np.sqrt(R_s * R_p)],
        ])

    @staticmethod
    def rotation(theta: float) -> np.ndarray:
        """坐标旋转 Mueller 矩阵

        Args:
            theta: 旋转角度 [rad]
        """
        c, s = np.cos(2 * theta), np.sin(2 * theta)
        return np.array([
            [1, 0, 0, 0],
            [0, c, s, 0],
            [0, -s, c, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def attenuation(transmittance: float) -> np.ndarray:
        """衰减 Mueller 矩阵"""
        return np.eye(4) * transmittance
