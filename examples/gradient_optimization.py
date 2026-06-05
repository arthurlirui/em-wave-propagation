"""
可微电磁传播 — 梯度优化示例

演示如何使用 PyTorch 自动求导进行：
1. 材料参数反演（已知路径损耗，反推材料折射率）
2. 发射源位置优化（给定接收点，优化 TX 位置）
3. RIS 反射相位优化

所有计算均支持自动微分，可直接反向传播梯度。
"""

from __future__ import annotations

import os
import sys

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propagation.diff_fresnel import (
    fresnel_reflection_coefficients,
    fresnel_reflectance,
    path_loss_fresnel,
    brewster_angle_approx,
)
from src.propagation.diff_utd import (
    utd_diffraction_coefficient_diff,
    diffraction_loss_db,
)
from src.propagation.diff_path import (
    total_path_loss,
    friis_free_space_loss,
    optimize_material_parameters,
    optimize_source_position,
)
from src.utils.constants import C, wavelength


def demo_fresnel_gradient():
    """Fresnel 系数梯度计算演示"""
    print("\n" + "=" * 60)
    print("  Demo 1: Fresnel 系数梯度")
    print("=" * 60)

    # 需要求解导数的变量
    n2_real = torch.tensor(4.5, requires_grad=True)  # 混凝土
    n2_imag = torch.tensor(-0.3, requires_grad=True)
    theta = torch.tensor(0.5, requires_grad=True)     # ~28.6°
    n1 = torch.tensor(1.0 + 0.0j)

    n2 = torch.complex(n2_real, n2_imag)
    r_te, r_tm = fresnel_reflection_coefficients(n1, n2, theta)

    # 计算梯度: 反射系数幅度 vs 折射率
    r_te_mag = torch.abs(r_te)
    r_te_mag.backward()

    print(f"\n  TE 反射系数幅度 |r_TE| = {r_te_mag.item():.4f}")
    print(f"  梯度 ∂|r_TE|/∂n'  = {n2_real.grad.item():.4f}")
    print(f"  梯度 ∂|r_TE|/∂n'' = {n2_imag.grad.item():.4f}")
    print(f"  梯度 ∂|r_TE|/∂θ   = {theta.grad.item():.4f}")
    print(f"\n  → 这告诉我们: n 每增加 1, |r_TE| 变化 {n2_real.grad.item():.4f}")


def demo_material_inversion():
    """材料参数反演 — 从路径损耗反推材料"""
    print("\n" + "=" * 60)
    print("  Demo 2: 材料参数反演")
    print("=" * 60)

    # 模拟"真实"材料（混凝土）
    n2_true = torch.complex(torch.tensor(5.3), torch.tensor(-0.34))
    n1 = torch.tensor(1.0 + 0.0j)
    theta = torch.tensor(0.4)
    true_loss = path_loss_fresnel(n1, n2_true, theta, 1)
    print(f"  '真实'混凝土: n₂ = 5.30 - j0.34, 损耗 = {true_loss.item():.2f} dB")

    # 从损耗反推材料
    n2_opt, history = optimize_material_parameters(
        target_loss_db=true_loss.item(),
        freq=28e9,
        los_distance=10.0,
        theta_i=0.4,
        num_reflections=1,
        n1_real=1.0,
        num_steps=200,
    )

    error = torch.abs(n2_opt - n2_true).item()
    print(f"  与真实值误差: {error:.4f}")


def demo_source_optimization():
    """发射源位置优化"""
    print("\n" + "=" * 60)
    print("  Demo 3: 发射源位置优化")
    print("=" * 60)

    rx = torch.tensor([8.0, 1.0, 1.5])
    tx_opt = optimize_source_position(
        rx_position=rx,
        desired_power_db=-55.0,
        freq=28e9,
        n2=5.3 - 0.34j,  # 混凝土地面
        num_steps=100,
    )


def demo_diffraction_gradient():
    """UTD 绕射系数梯度计算"""
    print("\n" + "=" * 60)
    print("  Demo 4: UTD 绕射系数梯度")
    print("=" * 60)

    # 可微参数
    freq = torch.tensor(28e9, requires_grad=True)
    phi = torch.tensor(0.8, requires_grad=True)        # 观察角
    phi_prime = torch.tensor(0.5, requires_grad=True)   # 入射角
    n_factor = torch.tensor(1.5, requires_grad=True)    # 楔形因子

    k = 2 * torch.pi * freq / C

    D = utd_diffraction_coefficient_diff(k, phi, phi_prime,
                                          torch.tensor(torch.pi/3), n_factor)
    D_mag = torch.abs(D)
    D_mag.backward()

    print(f"\n  |D| = {D_mag.item():.6f}")
    print(f"  ∂|D|/∂f  = {freq.grad.item():.6e} (∂/∂Hz)")
    print(f"  ∂|D|/∂φ  = {phi.grad.item():.6f}")
    print(f"  ∂|D|/∂φ' = {phi_prime.grad.item():.6f}")
    print(f"  ∂|D|/∂n  = {n_factor.grad.item():.6f}")


def demo_ris_phase():
    """RIS 反射相位优化（简化版）

    模拟可调智能表面的单个单元，优化反射相位以最大化接收功率。
    """
    print("\n" + "=" * 60)
    print("  Demo 5: RIS 相位优化")
    print("=" * 60)

    # RIS 单元模型: 反射幅度固定，相位可调
    phase = torch.tensor(0.0, requires_grad=True)  # 初始相位
    freq = torch.tensor(28e9)
    k = 2 * torch.pi * freq / C

    # 简化: 直接路径 + RIS 反射路径
    d_direct = 15.0  # 直达距离
    d_ris = 10.0      # TX→RIS + RIS→RX 距离

    optimizer = torch.optim.Adam([phase], lr=0.1)
    print(f"\n  步骤 |  接收功率 [dBm] | 相移 [rad] | 相移 [°]")

    for step in range(50):
        optimizer.zero_grad()

        # 直达路径
        e_direct = torch.exp(-1j * k * d_direct)

        # RIS 反射路径（幅度 0.8， 相位可调）
        ris_gain = 0.8
        ris_phase_shift = torch.exp(1j * phase)
        e_ris = ris_gain * ris_phase_shift * torch.exp(-1j * k * d_ris)

        # 合成场
        e_total = e_direct + e_ris
        power_db = 10 * torch.log10(torch.abs(e_total) ** 2 + 1e-15)

        # 最大化功率
        loss = -power_db
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 49:
            phase_deg = (phase.item() * 180 / torch.pi) % 360
            print(f"  {step:5d} | {power_db.item():>16.2f} | {phase.item():>10.4f} | {phase_deg:>7.1f}°")

    print(f"\n  ✅ 最优相移: {(phase.item() * 180 / torch.pi) % 360:.1f}°")
    print(f"     (理论最优: 使 e_direct 与 e_RIS 同相)")


def demo_sweep():
    """频段扫描 — 不同频率下的材料响应"""
    print("\n" + "=" * 60)
    print("  Demo 6: 频段扫描 (10-300 GHz)")
    print("=" * 60)

    n1 = torch.tensor(1.0 + 0.0j)
    theta = torch.tensor(0.4)  # ~23°

    print(f"\n  频率 [GHz] | 混凝土 n' | 混凝土 n'' | 反射损耗 [dB] | Brewster 角")
    print(f"  " + "-" * 65)

    for f_ghz in [10, 28, 60, 100, 200, 300]:
        freq = f_ghz * 1e9
        # 使用 Debye 模型（从 materials.base 获取）
        from src.materials.base import get_material
        concrete = get_material("concrete")
        eps = concrete.permittivity(freq)
        n = np.sqrt(eps)

        # Fresnel 计算
        loss = path_loss_fresnel(
            n1,
            torch.tensor(complex(n), dtype=torch.complex64),
            theta,
            1,
        )
        theta_b = brewster_angle_approx(
            n1,
            torch.tensor(complex(n), dtype=torch.complex64),
        )

        print(f"  {f_ghz:>8.0f}   | {eps.real:>8.2f}    | {abs(eps.imag):>8.4f}   | "
              f"{loss.item():>12.2f}   | {theta_b.item()*180/torch.pi:>7.1f}°")


def main():
    print("=" * 60)
    print("  可微电磁传播仿真 — 梯度优化")
    print("  Differentiable EM Propagation")
    print("=" * 60)

    print(f"\n  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    demo_fresnel_gradient()
    demo_material_inversion()
    demo_source_optimization()
    demo_diffraction_gradient()
    demo_ris_phase()
    demo_sweep()

    print("\n" + "=" * 60)
    print("  全部演示完成 ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
