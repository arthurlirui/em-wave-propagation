"""
3D 可视化模块

使用 matplotlib 渲染：
- 3D 场景网格
- 射线传播路径
- 场强覆盖图
- 极化态
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.ray import PathSegment, PropagationPath, Source, Receiver
from src.geometry.primitive import SceneObject, Triangle


# ── matplotlib 导入（带 fallback） ────────────────────────────────

HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MPL = True
except ImportError:
    plt = None


class Arrow3D(FancyArrowPatch):
    """3D 箭头（用于射线方向标注）"""
    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._posA = posA
        self._posB = posB

    def do_3d_projection(self):
        xs, ys, zs = proj_transform(self._posA, self._posB, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def plot_scene(
    objects: List[SceneObject],
    ax=None,
    color_map: Optional[Dict[str, str]] = None,
    alpha: float = 0.6,
) -> None:
    """在 3D 坐标轴上绘制场景网格

    Args:
        objects: 场景对象列表
        ax: matplotlib 3D 坐标轴 (None = 新建)
        color_map: 材质 → 颜色 映射
        alpha: 透明度
    """
    if not HAS_MPL:
        print("⚠️ matplotlib not installed. pip install matplotlib")
        return

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

    default_colors = {
        "concrete": "#888888",
        "drywall": "#d4c9b0",
        "glass": "#87ceeb",
        "metal": "#c0c0c0",
        "wood": "#8b6914",
        "vacuum": "#ffffff",
    }
    color_map = color_map or {}

    for obj in objects:
        for tri in obj.triangles:
            color = color_map.get(tri.material, default_colors.get(tri.material, "#aaaaaa"))
            verts = tri.vertices
            poly = Poly3DCollection([verts], alpha=alpha, color=color, edgecolor="black", linewidth=0.3)
            ax.add_collection3d(poly)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_box_aspect([1, 1, 1])


def plot_rays(
    paths_by_rx: Dict[int, List[PropagationPath]],
    ax=None,
    source: Optional[Source] = None,
    receivers: Optional[List[Receiver]] = None,
    max_paths: int = 50,
    colormap_name: str = "viridis",
) -> None:
    """在 3D 坐标轴上绘制射线传播路径

    Args:
        paths_by_rx: 按接收点索引分组的传播路径
        ax: matplotlib 3D 坐标轴
        source: 发射源（绘制标记）
        receivers: 接收点列表（绘制标记）
        max_paths: 最大绘制的路径数
        colormap_name: 路径按损耗着色的 colormap
    """
    if not HAS_MPL:
        return

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap(colormap_name)

    # 收集所有路径及其损耗
    all_paths = []
    for rx_idx, paths in paths_by_rx.items():
        for path in paths:
            all_paths.append((rx_idx, path))

    if not all_paths:
        print("⚠️ No paths to plot")
        return

    # 按损耗排序，取前 N 条
    all_paths.sort(key=lambda x: x[1].total_loss_db)

    if max_paths > 0:
        all_paths = all_paths[:max_paths]

    # 归一化损耗用于着色
    losses = [p.total_loss_db for _, p in all_paths]
    min_loss, max_loss = min(losses), max(losses)
    loss_range = max_loss - min_loss if max_loss > min_loss else 1.0

    for rx_idx, path in all_paths:
        norm_loss = (path.total_loss_db - min_loss) / loss_range
        color = cmap(1.0 - norm_loss)

        for seg in path.segments:
            xs = [seg.start[0], seg.end[0]]
            ys = [seg.start[1], seg.end[1]]
            zs = [seg.start[2], seg.end[2]]
            ax.plot(xs, ys, zs, color=color, alpha=0.7, linewidth=1.5)

    # 绘制发射源
    if source is not None:
        ax.scatter(*source.position, color="red", s=200, marker="o", label="Source")
        ax.text(*source.position, f"TX\n{source.power_dbm} dBm", fontsize=9)

    # 绘制接收点
    if receivers is not None:
        for i, rx in enumerate(receivers):
            ax.scatter(*rx.position, color="blue", s=100, marker="^", label=f"RX{i}" if i == 0 else "")
            ax.text(*rx.position, f"RX{i}", fontsize=9)

    ax.legend()


def plot_field_strength_2d(
    field_map: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_slice: float = 1.5,
    title: str = "Field Strength [dBm]",
    ax=None,
):
    """绘制 2D 场强覆盖切片图

    Args:
        field_map: (nx, ny) 场强矩阵 [dBm]
        x_range: (x_min, x_max) [m]
        y_range: (y_min, y_max) [m]
        z_slice: 切片高度 [m]
        title: 图标题
        ax: matplotlib 坐标轴
    """
    if not HAS_MPL:
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    im = ax.imshow(field_map.T, extent=extent, origin="lower",
                   cmap="hot", aspect="auto")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"{title} @ z={z_slice}m")
    plt.colorbar(im, ax=ax, label="Power [dBm]")


def plot_coverage_3d(
    objects: List[SceneObject],
    paths_by_rx: Dict[int, List[PropagationPath]],
    source: Optional[Source] = None,
    receivers: Optional[List[Receiver]] = None,
    show_polarization: bool = False,
) -> None:
    """完整 3D 可视化：场景 + 射线路径 + 标记

    这是最高层次的绘图函数，一键生成完整可视化。

    Args:
        objects: 场景对象列表
        paths_by_rx: 路径数据
        source: 发射源
        receivers: 接收点列表
        show_polarization: 是否显示偏振信息
    """
    if not HAS_MPL:
        print("⚠️ matplotlib not available. Run: pip install matplotlib")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制场景
    plot_scene(objects, ax=ax, alpha=0.4)

    # 绘制路径
    plot_rays(paths_by_rx, ax=ax, source=source, receivers=receivers)

    # 调整坐标范围
    all_verts = []
    for obj in objects:
        for tri in obj.triangles:
            all_verts.extend(tri.vertices)
    if all_verts:
        verts_arr = np.array(all_verts)
        min_b = verts_arr.min(axis=0)
        max_b = verts_arr.max(axis=0)
        padding = np.max(max_b - min_b) * 0.1
        ax.set_xlim(min_b[0] - padding, max_b[0] + padding)
        ax.set_ylim(min_b[1] - padding, max_b[1] + padding)
        ax.set_zlim(min_b[2] - padding, max_b[2] + padding)

    ax.set_title("EM Wave Propagation — 3D Ray Tracing", fontsize=14)
    plt.tight_layout()
    plt.show()
