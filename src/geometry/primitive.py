"""
场景图与几何基元
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Triangle:
    """三角面元"""
    vertices: np.ndarray  # (3, 3) 顶点坐标
    material: str = "vacuum"
    normal: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.normal is None:
            v0, v1, v2 = self.vertices
            self.normal = np.cross(v1 - v0, v2 - v0)
            self.normal /= np.linalg.norm(self.normal) + 1e-15

    def centroid(self) -> np.ndarray:
        return np.mean(self.vertices, axis=0)

    def area(self) -> float:
        v0, v1, v2 = self.vertices
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))


@dataclass
class Edge:
    """场景边缘（用于 UTD 绕射）"""
    start: np.ndarray
    end: np.ndarray
    face0_material: str = "vacuum"
    face1_material: str = "vacuum"
    wedge_angle: float = np.pi  # 楔角 [rad] (默认 180° = 半平面)

    @property
    def direction(self) -> np.ndarray:
        d = self.end - self.start
        return d / (np.linalg.norm(d) + 1e-15)

    @property
    def length(self) -> float:
        return np.linalg.norm(self.end - self.start)

    @property
    def n_factor(self) -> float:
        """UTD 楔形因子 n (nπ = 楔角)"""
        return self.wedge_angle / np.pi


@dataclass
class SceneObject:
    """场景对象"""
    name: str
    triangles: List[Triangle] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def add_box(self, center, size, material: str = "concrete"):
        """添加长方体"""
        center = np.asarray(center, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        x, y, z = center
        dx, dy, dz = size / 2
        verts = np.array([
            [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz], [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
            [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz], [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz],
        ])
        # 12 个三角形组成 6 个面
        faces = [
            (0,1,2), (0,2,3),  # 底
            (4,6,5), (4,7,6),  # 顶
            (0,4,5), (0,5,1),  # 前
            (2,6,7), (2,7,3),  # 后
            (0,3,7), (0,7,4),  # 左
            (1,5,6), (1,6,2),  # 右
        ]
        for f in faces:
            self.triangles.append(Triangle(
                vertices=verts[list(f)],
                material=material,
            ))

    def add_edge(self, start: np.ndarray, end: np.ndarray,
                 material0: str = "concrete", material1: str = "concrete",
                 wedge_angle: float = np.pi):
        self.edges.append(Edge(start, end, material0, material1, wedge_angle))
