"""
示例：室内场景毫米波传播

模拟 28 GHz 室内办公场景，计算接收信号覆盖图。
"""

import math
import numpy as np

# 将项目根加入 path
import sys
sys.path.insert(0, ".")

from src.core.ray import Source, Receiver
from src.geometry.primitive import SceneObject
from src.propagation.sbr import SBRConfig, SBREngine
from src.materials.base import Material, get_material, list_materials


def build_indoor_scene() -> list[SceneObject]:
    """构建室内办公场景

    房间 10m × 8m × 3m，内有隔墙和金属柜。
    """
    objects = []

    # 房间（四面墙 + 天花板 + 地板）
    room = SceneObject("room")
    # 地板
    room.add_box([5, -0.1, 1.5], [10, 0.1, 3], "concrete")
    # 天花板
    room.add_box([5, 3.1, 1.5], [10, 0.1, 3], "drywall")
    # 后墙
    room.add_box([-0.1, 1.5, 1.5], [0.1, 3, 3], "drywall")
    # 前墙
    room.add_box([10.1, 1.5, 1.5], [0.1, 3, 3], "drywall")
    # 左墙
    room.add_box([5, 1.5, -0.1], [10, 3, 0.1], "drywall")
    # 右墙
    room.add_box([5, 1.5, 3.1], [10, 3, 0.1], "drywall")
    objects.append(room)

    # 隔墙（中间）
    wall = SceneObject("partition")
    wall.add_box([5, 1.5, 1.5], [0.1, 2.8, 3], "drywall")
    objects.append(wall)

    # 金属柜
    cabinet = SceneObject("cabinet")
    cabinet.add_box([2, 0.5, 2.5], [0.8, 1, 0.6], "metal")
    objects.append(cabinet)

    return objects


def main():
    print("=" * 60)
    print("  EM 波传播仿真 — 室内 28 GHz 覆盖图")
    print("=" * 60)

    # 可用材料
    print(f"\n可用材料: {list_materials()}")

    # 混凝土在 28 GHz 的介电常数
    concrete = get_material("concrete")
    eps = concrete.permittivity(28e9)
    n = concrete.refractive_index(28e9)
    print(f"\n混凝土 @ 28 GHz:")
    print(f"  ε_r = {eps.real:.2f} - j{abs(eps.imag):.4f}")
    print(f"  n   = {n.real:.3f} - j{abs(n.imag):.4f}")
    print(f"  反射损耗 @ 30° = {concrete.reflection_loss_db(28e9, math.radians(30)):.2f} dB")

    # 玻璃在 100 GHz 的介电常数
    glass = get_material("glass")
    eps_g = glass.permittivity(100e9)
    n_g = glass.refractive_index(100e9)
    print(f"\n玻璃 @ 100 GHz:")
    print(f"  ε_r = {eps_g.real:.2f} - j{abs(eps_g.imag):.4f}")

    # 构建场景
    print("\n构建场景...")
    scene_objects = build_indoor_scene()
    total_triangles = sum(len(obj.triangles) for obj in scene_objects)
    print(f"  三角面数: {total_triangles}")

    # 配置 SBR
    config = SBRConfig(
        num_rays=5000,          # 减少射线数加快演示
        max_bounces=3,
        frequency=28e9,
        receiver_radius=0.3,
    )

    engine = SBREngine(config)
    engine.load_scene(scene_objects)

    # 设置发射源
    source = Source(
        position=np.array([1.0, 1.0, 1.5]),
        frequency=28e9,
        power_dbm=20,  # 100 mW
        antenna_gain=5,  # 5 dBi
    )

    # 设置多个接收点
    receivers = [
        Receiver(position=np.array([3.0, 1.0, 1.5])),
        Receiver(position=np.array([5.0, 1.0, 1.5])),  # 墙后
        Receiver(position=np.array([7.0, 1.0, 2.0])),
        Receiver(position=np.array([9.0, 2.0, 1.5])),
    ]

    print(f"\n发射源: {source.position}")
    print(f"接收点数: {len(receivers)}")
    print(f"射线数: {config.num_rays}")
    print(f"最大弹射: {config.max_bounces}")
    print()

    # 执行 SBR
    print("执行射线发射...")
    paths_by_rx = engine.launch(source, receivers)

    # 输出结果
    print("\n" + "=" * 60)
    print("  结果")
    print("=" * 60)
    for rx_idx, paths in paths_by_rx.items():
        rx = receivers[rx_idx]
        print(f"\n接收点 {rx_idx} @ {rx.position}:")
        if paths:
            best_path = min(paths, key=lambda p: p.total_loss_db)
            print(f"  到达路径数: {len(paths)}")
            print(f"  最佳路径损耗: {best_path.total_loss_db:.2f} dB")
            print(f"  传播距离: {best_path.total_length:.2f} m")
            print(f"  时延: {best_path.delay * 1e9:.2f} ns")
        else:
            print(f"  无到达路径（屏蔽区）")


if __name__ == "__main__":
    main()
