"""
可微传播路径损耗

组合 Fresnel 反射 + UTD 绕射 + Friis 自由空间损耗，
所有路径在设计参数下完全可微。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from src.propagation.diff_fresnel import (
    fresnel_reflection_coefficients,
    fresnel_reflectance,
    path_loss_fresnel,
)
from src.propagation.diff_utd import (
    utd_diffraction_coefficient_diff,
    diffraction_loss_db,
)
from src.utils.constants import C


def friis_free_space_loss(freq: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Friis 自由空间损耗 [dB] — 可微

    FSPL = 20·log₁₀(4π·d·f / c)

    Args:
        freq: 频率 [Hz]（可微分）
        distance: 传播距离 [m]（可微分）

    Returns:
        loss_db: 自由空间损耗 [dB]
    """
    c = torch.tensor(C, dtype=freq.dtype)
    # 4π·d·f / c
    ratio = 4.0 * torch.pi * distance * freq / c
    fspl = 20.0 * torch.log10(ratio + 1e-15)
    return fspl


def total_path_loss(
    freq: torch.Tensor,
    los_distance: torch.Tensor,
    reflections: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    diffractions: Optional[List[Tuple[torch.Tensor, ...]]] = None,
    n1: torch.Tensor = torch.tensor(1.0 + 0.0j),
) -> torch.Tensor:
    """完整路径总损耗 [dB] — 完全可微

    组成:
        Loss_total = FSPL + Σ(反射损耗) + Σ(绕射损耗)

    Args:
        freq: 频率 [Hz]
        los_distance: LOS 传播总距离 [m]
        reflections: 反射列表, 每项 = (n2, theta_i, num_reflections)
            其中 n2 是反射介质的复折射率（优化目标）
        diffractions: 绕射列表, 每项 = (phi, phi_prime, beta0, n_factor, distance)
        n1: 入射介质折射率（默认空气=1）

    Returns:
        loss_db: 总路径损耗 [dB]（可微分）
    """
    # 1. 自由空间损耗
    loss = friis_free_space_loss(freq, los_distance)

    # 2. 反射损耗
    if reflections:
        for n2, theta_i, n_ref in reflections:
            loss = loss + path_loss_fresnel(n1, n2, theta_i, n_ref)

    # 3. 绕射损耗
    if diffractions:
        k = 2.0 * torch.pi * freq / c
        for phi, phi_prime, beta0, n_factor, dist in diffractions:
            loss = loss + diffraction_loss_db(k, phi, phi_prime, beta0, n_factor, dist)

    return loss


def optimize_material_parameters(
    target_loss_db: float,
    freq: float = 28e9,
    los_distance: float = 10.0,
    theta_i: float = 0.3,
    num_reflections: int = 1,
    n1_real: float = 1.0,
    learning_rate: float = 0.1,
    num_steps: int = 200,
) -> Tuple[torch.Tensor, List[float]]:
    """材料参数优化示例

    已知目标路径损耗，反推材料复折射率 n₂。

    Args:
        target_loss_db: 目标路径损耗 [dB]
        freq: 频率 [Hz]
        los_distance: LOS 距离 [m]
        theta_i: 入射角 [rad]
        num_reflections: 反射次数
        n1_real: 入射介质折射率实部
        learning_rate: 学习率
        num_steps: 优化步数

    Returns:
        n2_opt: 优化后的复折射率
        loss_history: 损耗历史
    """
    # 初始猜测
    n2_real = torch.tensor(3.0, requires_grad=True)
    n2_imag = torch.tensor(-0.1, requires_grad=True)

    n1 = torch.tensor(n1_real + 0.0j)
    freq_t = torch.tensor(freq, requires_grad=False)
    dist_t = torch.tensor(los_distance)
    theta_t = torch.tensor(theta_i, requires_grad=True)

    # Adam 优化器
    optimizer = torch.optim.Adam([n2_real, n2_imag, theta_t], lr=learning_rate)
    loss_history = []

    print(f"\n{'='*60}")
    print(f"  材料参数优化")
    print(f"{'='*60}")
    print(f"  目标损耗: {target_loss_db:.1f} dB")
    print(f"  频率: {freq/1e9:.1f} GHz")
    print(f"  距离: {los_distance:.1f} m")
    print(f"  入射角: {theta_i:.2f} rad")
    print(f"  初始猜测: n₂ = 3.0 - j0.1")
    print(f"\n  步骤 | 损耗 [dB] | n₂' | n₂'' | θ [rad]")

    for step in range(num_steps):
        optimizer.zero_grad()

        # 当前复折射率
        n2 = torch.complex(n2_real, n2_imag)
        loss_db = path_loss_fresnel(n1, n2, theta_t, num_reflections)

        # MSE 损失
        loss = (loss_db - target_loss_db) ** 2
        loss.backward()

        optimizer.step()

        if step % 20 == 0 or step == num_steps - 1:
            n2_opt = torch.complex(n2_real, n2_imag)
            print(f"  {step:5d} | {loss_db.item():>8.2f} | "
                  f"{n2_real.item():>6.3f} | {n2_imag.item():>7.4f} | {theta_t.item():.4f}")
            loss_history.append(loss_db.item())

    n2_opt = torch.complex(n2_real, n2_imag)
    print(f"\n  ✅ 最终 n₂ = {n2_opt.item():.4f}")
    print(f"     最终损耗: {loss_db.item():.2f} dB (目标: {target_loss_db:.1f} dB)")

    return n2_opt, loss_history


def optimize_source_position(
    rx_position: torch.Tensor,
    desired_power_db: float = -50.0,
    freq: float = 28e9,
    n2: complex = 3.0 - 0.1j,
    num_steps: int = 100,
) -> torch.Tensor:
    """发射源位置优化

    给定接收点位置和期望接收功率，反向优化发射源位置。

    Args:
        rx_position: 接收点位置 [x, y, z]
        desired_power_db: 期望接收功率 [dBm]
        freq: 频率
        n2: 反射介质复折射率
        num_steps: 优化步数

    Returns:
        tx_opt: 优化后的发射源位置
    """
    # 发射源位置初始猜测
    tx_pos = torch.tensor([1.0, 1.0, 1.5], requires_grad=True)
    optimizer = torch.optim.Adam([tx_pos], lr=0.05)

    n1 = torch.tensor(1.0 + 0.0j)
    n2_t = torch.tensor(complex(n2), dtype=torch.complex64)
    freq_t = torch.tensor(freq)

    print(f"\n{'='*60}")
    print(f"  发射源位置优化")
    print(f"{'='*60}")
    print(f"  接收点: ({rx_position[0]:.1f}, {rx_position[1]:.1f}, {rx_position[2]:.1f})")
    print(f"  目标功率: {desired_power_db:.1f} dBm")
    print(f"  材料: n₂ = {n2}")
    print(f"\n  步骤 |   损耗 [dB] |     TX位置")

    for step in range(num_steps):
        optimizer.zero_grad()

        # LOS 距离
        delta = rx_position - tx_pos
        distance = torch.norm(delta)
        theta_i = torch.acos(
            torch.abs(delta[1]) / (distance + 1e-15)
        )

        # 总损耗 = FSPL + 一次地面反射
        fspl = friis_free_space_loss(freq_t, distance)
        ref_loss = path_loss_fresnel(n1, n2_t, theta_i, 1)
        total_loss = fspl + ref_loss

        # 期望功率 = 发射功率 - 总损耗
        tx_power = torch.tensor(20.0)  # 20 dBm
        rx_power = tx_power - total_loss

        loss = (rx_power - desired_power_db) ** 2
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == num_steps - 1:
            print(f"  {step:5d} | {total_loss.item():>8.2f} | "
                  f"({tx_pos[0].item():.2f}, {tx_pos[1].item():.2f}, {tx_pos[2].item():.2f})")

    print(f"\n  ✅ 最终 TX 位置: ({tx_pos[0].item():.2f}, {tx_pos[1].item():.2f}, {tx_pos[2].item():.2f})")
    print(f"     RX 位置: ({rx_position[0]:.1f}, {rx_position[1]:.1f}, {rx_position[2]:.1f})")
    print(f"     LOS 距离: {torch.norm(rx_position - tx_pos).item():.2f} m")

    return tx_pos
