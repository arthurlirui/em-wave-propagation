"""
射线发射引擎 (SBR — Shooting and Bouncing Rays)

核心算法：从发射源向场景发射密集射线束，
追踪每条射线的传播路径（反射/透射/绕射），
收集所有到达接收球的有效路径。

参考 Mitsuba 的路径追踪架构，但针对 EM 传播做了适配。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.ray import Ray, Intersection, PathSegment, PropagationPath, Source, Receiver
from src.geometry.primitive import SceneObject, Triangle, Edge
from src.materials.base import Material, get_material
from src.propagation.fresnel import fresnel_coefficients, reflectivity
from src.propagation.utd import utd_diffraction_coefficient
from src.utils.constants import C, wavelength, wavenumber


@dataclass
class SBRConfig:
    """射线发射引擎配置"""
    num_rays: int = 100_000          # 发射射线总数
    max_bounces: int = 5             # 最大弹射次数
    receiver_radius: float = 0.1     # 接收球半径 [m]
    power_threshold_db: float = -200 # 功率阈值 [dB]
    enable_diffraction: bool = True  # 是否启用 UTD 绕射
    enable_transmission: bool = True # 是否启用透射
    frequency: float = 28e9          # 默认频率 [Hz]


class SBREngine:
    """SBR 射线发射引擎"""

    def __init__(self, config: Optional[SBRConfig] = None):
        self.config = config or SBRConfig()
        self.objects: List[SceneObject] = []
        self._triangles: List[Triangle] = []
        self._edges: List[Edge] = []

    def add_object(self, obj: SceneObject):
        """添加场景对象"""
        self.objects.append(obj)
        self._triangles.extend(obj.triangles)
        self._edges.extend(obj.edges)

    def load_scene(self, objects: List[SceneObject]):
        """批量加载场景对象"""
        for obj in objects:
            self.add_object(obj)

    def _trace_ray(self, ray: Ray) -> List[PropagationPath]:
        """追踪单条射线，返回所有到达接收端的路径

        Args:
            ray: 初始射线

        Returns:
            到达接收端的所有传播路径
        """
        # ... 递归/迭代追踪射线路径
        # 此处为简化实现，复杂的 BVH 求交和路径收集在完整版本中
        return []

    def launch(self, source: Source, receivers: List[Receiver]) -> Dict[int, List[PropagationPath]]:
        """发射射线束并收集所有接收点的路径

        使用均匀球面发射 (HEALPix / Fibonacci 球面) 或经纬度采样。

        Args:
            source: 发射源
            receivers: 接收点列表

        Returns:
            receiver_index → [paths] 的映射
        """
        paths_by_rx: Dict[int, List[PropagationPath]] = {
            i: [] for i in range(len(receivers))
        }

        # 生成球面发射方向 (Fibonacci 球面)
        directions = self._generate_directions(self.config.num_rays)

        for i, direction in enumerate(directions):
            if i % 10000 == 0 and i > 0:
                print(f"  射线 {i}/{self.config.num_rays}")

            ray = Ray(
                origin=source.position.copy(),
                direction=direction,
                frequency=self.config.frequency,
                power=10**(source.power_dbm / 10) * 1e-3,  # dBm → W
                max_depth=self.config.max_bounces,
            )

            # 追踪该射线 → 收集路径
            self._trace_single_ray(ray, source, receivers, paths_by_rx)

        return paths_by_rx

    def _generate_directions(self, n: int) -> np.ndarray:
        """使用 Fibonacci 球面生成均匀分布的射线方向"""
        directions = np.zeros((n, 3))
        phi = math.pi * (3 - math.sqrt(5))  # 黄金角
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            directions[i] = [radius * math.cos(theta), y, radius * math.sin(theta)]
        return directions

    def _trace_single_ray(
        self,
        ray: Ray,
        source: Source,
        receivers: List[Receiver],
        paths_by_rx: Dict[int, List[PropagationPath]],
    ):
        """追踪单条射线经过的全部弹射"""

        current_ray = ray
        path = PropagationPath(frequency=ray.frequency)

        for bounce in range(self.config.max_bounces + 1):
            # 找最近交点
            hit, dist = self._find_nearest_intersection(current_ray)

            if hit is not None and dist > 1e-6:
                end_point = current_ray.propagate(dist)
                seg = PathSegment(
                    start=current_ray.origin.copy(),
                    end=end_point.copy(),
                    length=dist,
                )
                path.segments.append(seg)
                path.total_length += dist
            else:
                # 没有交点 → 射线射向无穷远, 仍检查接收
                end_point = current_ray.propagate(1000.0)  # 远场

            # 检查该段路径经过的所有接收点
            for rx_idx, rx in enumerate(receivers):
                if self._ray_passes_near_receiver(current_ray, rx, end_point, path):
                    new_path = PropagationPath(
                        segments=path.segments.copy(),
                        total_length=path.total_length,
                        frequency=ray.frequency,
                        power=current_ray.power,
                        total_loss_db=self._compute_path_loss(path, source, rx),
                    )
                    paths_by_rx[rx_idx].append(new_path)

            if hit is None or bounce == self.config.max_bounces:
                break

            # 处理表面交互
            if hit.is_edge and self.config.enable_diffraction:
                self._handle_diffraction(current_ray, hit, path)
            else:
                self._handle_reflection(current_ray, hit, path)

            # 功率低于阈值则终止
            if current_ray.power < 10**(self.config.power_threshold_db / 10):
                break

    def _ray_passes_near_receiver(
        self, ray: Ray, receiver: Receiver,
        segment_end: np.ndarray, path: PropagationPath
    ) -> bool:
        """检查射线是否经过接收球附近"""
        rx_pos = receiver.position
        r = self.config.receiver_radius

        # 方法1: 端点距离
        d_to_start = np.linalg.norm(ray.origin - rx_pos)
        d_to_end = np.linalg.norm(segment_end - rx_pos)

        # 方法2: 点到线段的最短距离
        v = segment_end - ray.origin
        w = rx_pos - ray.origin
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return d_to_start < r
        t = np.dot(w, v) / (v_norm * v_norm)
        t = max(0, min(1, t))
        closest = ray.origin + t * v
        d_to_segment = np.linalg.norm(closest - rx_pos)

        return d_to_segment < r

    def _find_nearest_intersection(self, ray: Ray) -> Tuple[Optional[Intersection], float]:
        """暴力求交（BVH 加速在完整版本中实现）

        遍历所有三角面，找最近的交点。
        """
        nearest_dist = float("inf")
        nearest_hit = None

        for tri_id, tri in enumerate(self._triangles):
            hit, dist = self._ray_triangle_intersect(ray, tri)
            if hit and dist > 1e-6 and dist < nearest_dist:
                nearest_dist = dist
                nearest_hit = Intersection(
                    point=ray.propagate(dist),
                    distance=dist,
                    normal=tri.normal,
                    material_name=tri.material,
                    primitive_id=tri_id,
                )

        return nearest_hit, nearest_dist

    def _ray_triangle_intersect(self, ray: Ray, tri: Triangle) -> Tuple[bool, float]:
        """Möller–Trumbore 射线-三角形求交算法"""
        v0, v1, v2 = tri.vertices
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        if abs(a) < 1e-10:
            return False, 0.0
        f = 1.0 / a
        s = ray.origin - v0
        u = f * np.dot(s, h)
        if u < 0 or u > 1:
            return False, 0.0
        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        if v < 0 or u + v > 1:
            return False, 0.0
        t = f * np.dot(edge2, q)
        return t > 1e-6, t

    def _handle_reflection(self, ray: Ray, hit: Intersection, path: PropagationPath):
        """处理 Fresnel 反射"""
        try:
            mat = get_material(hit.material_name)
        except KeyError:
            mat = get_material("vacuum")

        n1 = 1.0  # 空气
        n2 = mat.refractive_index(ray.frequency)

        cos_theta_i = abs(np.dot(ray.direction, hit.normal))
        theta_i = np.arccos(min(cos_theta_i, 1.0))

        r_te, r_tm, _, _ = fresnel_coefficients(n1, n2, theta_i)
        r_te_pow, r_tm_pow = reflectivity(r_te, r_tm)

        # 反射方向
        ray.direction = ray.direction - 2 * np.dot(ray.direction, hit.normal) * hit.normal
        ray.origin = hit.point + ray.direction * 1e-6
        ray.power *= (r_te_pow + r_tm_pow) / 2
        ray.depth += 1

    def _handle_diffraction(self, ray: Ray, hit: Intersection, path: PropagationPath):
        """处理 UTD 边缘绕射"""
        # 简化实现：仅将绕射射线沿 Keller 锥方向传播
        edge_dir = hit.edge_direction
        if edge_dir is None:
            return

        # 沿边缘切线方向继续传播
        ray.origin = hit.point + ray.direction * 1e-6
        ray.power *= 0.5  # 绕射效率因子
        ray.depth += 1

    def _compute_path_loss(self, path: PropagationPath, source: Source, rx: Receiver) -> float:
        """计算完整路径损耗 [dB]"""
        d = path.total_length
        f = path.frequency
        # Friis 自由空间损耗
        fspl = 20 * np.log10(4 * np.pi * d * f / C)
        # 反射/透射/绕射累积损耗
        loss = fspl - source.antenna_gain - rx.antenna_gain
        return loss
