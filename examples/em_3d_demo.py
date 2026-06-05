"""
完整 3D 电磁波传播模拟演示

加载 OBJ 三维场景 → 赋予电磁材料参数 → SBR 射线追踪 → 3D 可视化

波长范围: 0.1 mm (3 THz) ~ 10 cm (3 GHz)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# 项目根路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.ray import Source, Receiver, PropagationPath
from src.geometry.io import load_obj
from src.geometry.primitive import SceneObject
from src.materials.base import get_material, list_materials
from src.propagation.sbr import SBRConfig, SBREngine

try:
    from src.renderer.visualization import plot_coverage_3d, plot_scene, plot_rays
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

# ── 场景参数 ─────────────────────────────────────────────────────

# 电磁材质 → 3D 模型材质名 映射
MATERIAL_MAP = {
    "Wall": "drywall",
    "Floor": "concrete",
    "Ceiling": "drywall",
    "Glass": "glass",
    "Metal": "metal",
}

# 可视化颜色映射
COLOR_MAP = {
    "drywall": "#d4c9b0",
    "concrete": "#888888",
    "glass": "#87ceeb",
    "metal": "#c0c0c0",
}


def build_scene_from_obj(obj_path: str) -> Tuple[List[SceneObject], Dict]:
    """从 OBJ 文件加载场景，返回 (objects, metadata)"""
    print(f"📦 加载场景: {obj_path}")
    objects = load_obj(obj_path, material_map=MATERIAL_MAP)

    total_triangles = sum(len(o.triangles) for o in objects)
    print(f"   对象数: {len(objects)}")
    print(f"   三角面数: {total_triangles}")
    for obj in objects:
        print(f"     - {obj.name}: {len(obj.triangles)} 面, 材质={obj.triangles[0].material}")

    return objects, {"total_triangles": total_triangles}


def simulate(
    objects: List[SceneObject],
    frequency: float = 28e9,
    num_rays: int = 20000,
    max_bounces: int = 3,
) -> Tuple[Dict[int, List[PropagationPath]], Source, List[Receiver]]:
    """执行电磁传播模拟"""
    wavelength_m = 299_792_458 / frequency
    print(f"\n📡 模拟参数:")
    print(f"   频率: {frequency/1e9:.1f} GHz")
    print(f"   波长: {wavelength_m*1000:.2f} mm (λ)")
    print(f"   射线数: {num_rays}")
    print(f"   最大弹射: {max_bounces}")

    # 配置 SBR 引擎
    config = SBRConfig(
        num_rays=num_rays,
        max_bounces=max_bounces,
        frequency=frequency,
        receiver_radius=wavelength_m * 5,  # 接收球 = 5λ
    )
    engine = SBREngine(config)
    engine.load_scene(objects)

    # 发射源 (房间左下角)
    source = Source(
        position=np.array([1.0, 1.0, 1.5]),
        frequency=frequency,
        power_dbm=20.0,  # 100 mW
        antenna_gain=3.0,  # 3 dBi
    )

    # 接收点 (沿房间路径布置)
    receivers = [
        Receiver(position=np.array([3.0, 1.0, 1.5])),   # RX0: 近端 LOS
        Receiver(position=np.array([5.5, 1.0, 1.5])),   # RX1: 隔墙后 (NLOS)
        Receiver(position=np.array([8.0, 1.0, 1.5])),   # RX2: 远端
        Receiver(position=np.array([3.0, 2.0, 2.5])),   # RX3: 对角高位
        Receiver(position=np.array([9.0, 0.5, 0.5])),   # RX4: 远端角落
    ]

    print(f"\n🔴 发射源: ({source.position[0]:.1f}, {source.position[1]:.1f}, {source.position[2]:.1f})")
    print(f"   {source.power_dbm} dBm / {source.antenna_gain} dBi")
    print(f"\n🔵 接收点: {len(receivers)} 个")
    for i, rx in enumerate(receivers):
        dist = np.linalg.norm(rx.position - source.position)
        print(f"    RX{i}: ({rx.position[0]:.1f}, {rx.position[1]:.1f}, {rx.position[2]:.1f}) 距离={dist:.1f}m")

    # 执行 SBR
    print(f"\n🚀 执行射线发射 ({num_rays} 条)...")
    t_start = time.time()
    paths_by_rx = engine.launch(source, receivers)
    elapsed = time.time() - t_start

    # 输出结果
    print(f"\n⏱️  耗时: {elapsed:.2f}s")
    print(f"\n{'='*60}")
    print(f"  传播路径结果")
    print(f"{'='*60}")

    total_paths = 0
    for i, rx in enumerate(receivers):
        paths = paths_by_rx[i]
        total_paths += len(paths)
        if paths:
            best = min(paths, key=lambda p: p.total_loss_db)
            print(f"\n  📍 RX{i} @ ({rx.position[0]:.1f}, {rx.position[1]:.1f}, {rx.position[2]:.1f}):")
            print(f"      到达路径数: {len(paths)}")
            print(f"      最佳路径损耗: {best.total_loss_db:.1f} dB")
            print(f"      传播距离: {best.total_length:.2f} m")
            print(f"      时延: {best.delay*1e9:.1f} ns")
            if best.excess_loss_db > 0:
                print(f"      额外损耗: {best.excess_loss_db:.1f} dB (多径/反射)")
        else:
            print(f"\n  📍 RX{i} @ ({rx.position[0]:.1f}, {rx.position[1]:.1f}, {rx.position[2]:.1f}):")
            print(f"      ❌ 无到达路径 (阴影区)")

    print(f"\n总计: {total_paths} 条传播路径")

    return paths_by_rx, source, receivers


def visualize(objects, paths_by_rx, source, receivers, output_file: str = None):
    """3D 可视化"""
    if not HAS_VIZ:
        print("\n⚠️ 可视化需要 matplotlib。安装: pip install matplotlib")
        print("   显示文本报告代替可视化。")
        return

    print("\n🎨 生成 3D 可视化...")

    import matplotlib
    matplotlib.use("Agg")  # 无头模式
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 场景网格
    plot_scene(objects, ax=ax, color_map=COLOR_MAP, alpha=0.35)

    # 射线路径 (前 100 条)
    plot_rays(paths_by_rx, ax=ax, source=source, receivers=receivers, max_paths=100)

    # 标题
    freq_ghz = source.frequency / 1e9
    wavelength_mm = 299.792458 / source.frequency * 1000
    ax.set_title(
        f"EM Wave Propagation — {freq_ghz:.1f} GHz (λ={wavelength_mm:.2f} mm)\n"
        f"Source @ ({source.position[0]:.1f}, {source.position[1]:.1f}, {source.position[2]:.1f})",
        fontsize=13,
    )

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"   已保存: {output_file}")
    else:
        plt.show()


def print_report(objects, paths_by_rx, source, receivers):
    """打印文本报告（含物理参数明细）"""
    freq = source.frequency
    lam = 299_792_458 / freq

    print(f"\n{'='*70}")
    print(f"  电磁波传播模拟报告")
    print(f"{'='*70}")
    print(f"\n📡 系统参数:")
    print(f"  频率: {freq/1e9:.1f} GHz")
    print(f"  波长: {lam*1000:.2f} mm")
    print(f"  波数: {2*np.pi/lam:.2f} rad/m")
    print(f"  自由空间传播损耗 @ 1m: {20*np.log10(4*np.pi*1*freq/299792458):.1f} dB")

    print(f"\n🏗️  场景参数:")
    print(f"  对象数: {len(objects)}")
    total_tris = sum(len(o.triangles) for o in objects)
    print(f"  三角面数: {total_tris}")

    # 材料明细
    materials_used = set()
    for obj in objects:
        for tri in obj.triangles:
            materials_used.add(tri.material)
    print(f"  材料: {', '.join(materials_used)}")
    for mat_name in materials_used:
        try:
            mat = get_material(mat_name)
            eps = mat.permittivity(freq)
            print(f"    {mat_name}: ε_r = {eps.real:.2f} - j{abs(eps.imag):.4f}")
        except KeyError:
            pass

    # 路径统计
    all_paths = [p for paths in paths_by_rx.values() for p in paths]
    if all_paths:
        losses = [p.total_loss_db for p in all_paths]
        delays = [p.delay * 1e9 for p in all_paths]
        print(f"\n📊 路径统计:")
        print(f"  总路径数: {len(all_paths)}")
        print(f"  最小损耗: {min(losses):.1f} dB")
        print(f"  最大损耗: {max(losses):.1f} dB")
        print(f"  最大时延扩展: {max(delays)-min(delays):.1f} ns")


def main():
    print("=" * 70)
    print("  📡 高频电磁波自由空间传播 — 3D 精确模拟")
    print(f"  波长范围: 0.1 mm ~ 10 cm (3 GHz ~ 3 THz)")
    print("=" * 70)

    # 可用材料
    print(f"\n📋 可用电磁材料: {list_materials()}")
    for mat_name in list_materials():
        mat = get_material(mat_name)
        eps_100g = mat.permittivity(100e9)
        eps_300g = mat.permittivity(300e9)
        print(f"  {mat_name:15s}: ε_r(100GHz)={eps_100g.real:.2f} ε_r(300GHz)={eps_300g.real:.2f}")

    # 生成/加载 3D 场景
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    obj_path = os.path.join(models_dir, "indoor_office.obj")
    if not os.path.isfile(obj_path):
        print(f"\n🏗️  生成示例场景...")
        from examples.generate_scene import generate_office_scene
        generate_office_scene(models_dir)

    objects, meta = build_scene_from_obj(obj_path)

    # ===== 扫描不同频段 =====
    frequencies = [10e9, 28e9, 100e9, 300e9]  # 10 GHz ~ 300 GHz

    for freq in frequencies:
        lam = 299_792_458 / freq
        print(f"\n{'─'*70}")
        print(f"  {'═'*66}")
        print(f"  ▶ 频率: {freq/1e9:>6.0f} GHz  |  波长: {lam*1000:>6.2f} mm  |  "
              f"波数: {2*np.pi/lam:>8.1f} rad/m")
        print(f"  {'═'*66}")

        num_rays = 5000 if freq <= 28e9 else 10000
        paths_by_rx, source, receivers = simulate(
            objects,
            frequency=freq,
            num_rays=num_rays,
            max_bounces=3,
        )

    # 可视化 (只对 28 GHz 做)
    freq_main = 28e9
    lam_main = 299_792_458 / freq_main
    print(f"\n{'─'*70}")
    print(f"  ▶ 最终可视化: {freq_main/1e9} GHz, λ = {lam_main*1000:.2f} mm")
    paths_by_rx, source, receivers = simulate(
        objects, frequency=freq_main, num_rays=20000, max_bounces=3,
    )

    print_report(objects, paths_by_rx, source, receivers)
    visualize(objects, paths_by_rx, source, receivers,
              output_file=os.path.join(models_dir, "..", "simulation_result.png"))

    print(f"\n{'='*70}")
    print(f"  ✅ 模拟完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
