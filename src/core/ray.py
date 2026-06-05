"""
核心数据结构 — 射线、交点、传播路径
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Ray:
    """电磁射线

    Attributes:
        origin: 起点 [m]
        direction: 单位方向矢量
        frequency: 频率 [Hz]
        power: 功率 [W]
        phase: 相位 [rad]
        polarization: Stokes 矢量 [I, Q, U, V]
        depth: 已弹射次数
        max_depth: 最大弹射次数
    """
    origin: np.ndarray
    direction: np.ndarray
    frequency: float
    power: float = 1.0
    phase: float = 0.0
    polarization: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    depth: int = 0
    max_depth: int = 10
    segments: List["PathSegment"] = field(default_factory=list)

    @property
    def wavelength(self) -> float:
        return 299_792_458 / self.frequency

    @property
    def wavenumber(self) -> float:
        return 2 * np.pi * self.frequency / 299_792_458

    def propagate(self, distance: float) -> np.ndarray:
        """沿射线方向移动指定距离，返回新位置"""
        return self.origin + self.direction * distance


@dataclass
class Intersection:
    """射线-场景交点

    Attributes:
        point: 交点坐标 [m]
        distance: 从射线起点到交点的距离 [m]
        normal: 表面法向量
        material_name: 表面材料名称
        edge_direction: 如果是边缘，边缘方向向量
        is_edge: 是否边缘交点
    """
    point: np.ndarray
    distance: float
    normal: np.ndarray
    material_name: str = "vacuum"
    edge_direction: Optional[np.ndarray] = None
    is_edge: bool = False
    primitive_id: int = -1


@dataclass
class PathSegment:
    """传播路径段"""
    start: np.ndarray
    end: np.ndarray
    length: float
    interaction_type: str = "LOS"  # LOS | REFLECTION | TRANSMISSION | DIFFRACTION


@dataclass
class PropagationPath:
    """完整传播路径（从发射到接收的多段路径）"""
    segments: List[PathSegment] = field(default_factory=list)
    total_length: float = 0.0
    total_loss_db: float = 0.0
    power: float = 0.0
    phase: float = 0.0
    polarization: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    frequency: float = 10e9

    @property
    def delay(self) -> float:
        """传播时延 [s]"""
        return self.total_length / 299_792_458

    @property
    def excess_loss_db(self) -> float:
        """超出自由空间传播的额外损耗 [dB]"""
        from src.utils.constants import C
        fspl = 20 * np.log10(4 * np.pi * self.total_length * self.frequency / C)
        return self.total_loss_db - fspl


@dataclass
class Source:
    """发射源"""
    position: np.ndarray
    frequency: float
    power_dbm: float = 0.0  # [dBm]
    antenna_gain: float = 0.0  # [dBi]
    polarization: str = "vertical"  # vertical | horizontal | circular


@dataclass
class Receiver:
    """接收点"""
    position: np.ndarray
    sensitivity_dbm: float = -100.0
    antenna_gain: float = 0.0
    noise_figure: float = 6.0
