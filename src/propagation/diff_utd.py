"""
可微 UTD 绕射系数 — PyTorch Autograd

支持对楔角、入射角、频率求导，用于边缘绕射场景的参数优化。
"""

from __future__ import annotations

from typing import Tuple

import torch


def fresnel_transition_diff(x: torch.Tensor) -> torch.Tensor:
    """可微 Fresnel 过渡函数 F(x)

    F(x) = 2j√x · exp(jx) · ∫_{√x}^{∞} exp(-jτ²) dτ

    Args:
        x: 过渡参数（可微分）

    Returns:
        F(x): 复数 Fresnel 过渡函数
    """
    x = torch.clamp(x, min=1e-10)

    # 使用 PyTorch 的误差函数实现复数 Fresnel 积分
    # ∫ exp(jτ²) dτ = √(π/4) · erf(√(j) · τ) / √(j)

    sqrt_x = torch.sqrt(x)
    j = torch.tensor(1j, dtype=torch.complex64)

    # √(j) = exp(jπ/4) = 1/√2 + j/√2
    sqrt_j = torch.exp(j * torch.tensor(torch.pi / 4))

    # 复数误差函数
    # ∫_{√x}^{∞} exp(-jτ²) dτ
    # = √(π/4) · (1 - erf(√(j) · √x)) / √(-j)

    # 使用近似公式
    # F(x) ≈ exp(j(x+π/4)) / (2√(2π)x)  (大 x)
    # F(x) ≈ √(πx) - 2jx/3 · exp(jx)    (小 x)

    x_small = x < 0.5
    x_large = ~x_small

    F = torch.zeros_like(x, dtype=torch.complex64)

    if x_small.any():
        sqrt_x_small = sqrt_x[x_small]
        F_small = (torch.sqrt(torch.tensor(torch.pi) * x_small.float() * x_small.float())
                   - 2j * x[x_small] / 3 * torch.exp(j * x[x_small]))
        F[x_small] = F_small

    if x_large.any():
        F_large = torch.exp(j * (x[x_large] + torch.tensor(torch.pi / 4)))
        F_large = F_large / (2 * torch.sqrt(torch.tensor(2 * torch.pi) * x[x_large]))
        F[x_large] = F_large

    return F


def cot_safe(x: torch.Tensor) -> torch.Tensor:
    """安全的余切函数（避免极点）"""
    mask = torch.abs(x) < 1e-10
    x = torch.where(mask, torch.full_like(x, 1e-10), x)
    return 1.0 / torch.tan(x)


def utd_diffraction_coefficient_diff(
    k: torch.Tensor,
    phi: torch.Tensor,
    phi_prime: torch.Tensor,
    beta0: torch.Tensor,
    n_factor: torch.Tensor,
    r0: torch.Tensor = 0.0,
    rn: torch.Tensor = 0.0,
) -> torch.Tensor:
    """可微 UTD 绕射系数 Dₛ (硬边界)

    所有输入参数均支持 PyTorch 自动求导。

    Args:
        k: 波数 [rad/m] — 可微分（用频率优化时）
        phi: 观察角 [rad] — 可微分
        phi_prime: 入射角 [rad] — 可微分
        beta0: Keller 锥角 [rad] — 可微分
        n_factor: 楔形因子 (nπ = 楔角) — 可微分
        r0: 0-face 反射系数
        rn: n-face 反射系数

    Returns:
        D: 绕射系数 (complex) — 可微分
    """
    # 安全的余切
    def a_plus(beta):
        return 2.0 * torch.cos(
            (2 * torch.pi * n_factor * _N_plus(phi, phi_prime, beta, n_factor)
             - (phi - phi_prime)) / 2.0
        ) ** 2

    def a_minus(beta):
        return 2.0 * torch.cos(
            (2 * torch.pi * n_factor * _N_minus(phi, phi_prime, beta, n_factor)
             - (phi - phi_prime)) / 2.0
        ) ** 2

    L = 1.0 * torch.sin(beta0) ** 2  # 距离因子（简化：s=1）

    # 四项求和
    term1 = cot_safe((torch.pi + (phi - phi_prime)) / (2 * n_factor)) * \
            fresnel_transition_diff(k * L * a_plus(phi - phi_prime))
    term2 = cot_safe((torch.pi - (phi - phi_prime)) / (2 * n_factor)) * \
            fresnel_transition_diff(k * L * a_minus(phi - phi_prime))
    term3 = r0 * cot_safe((torch.pi - (phi + phi_prime)) / (2 * n_factor)) * \
            fresnel_transition_diff(k * L * a_minus(phi + phi_prime))
    term4 = rn * cot_safe((torch.pi + (phi + phi_prime)) / (2 * n_factor)) * \
            fresnel_transition_diff(k * L * a_plus(phi + phi_prime))

    # 前因子
    prefactor = -torch.exp(-1j * torch.pi / 4) / (
        2 * n_factor * torch.sqrt(2 * torch.pi * k) * torch.sin(beta0)
    )

    D = prefactor * (term1 + term2 + term3 + term4)
    return D


def _N_plus(
    phi: torch.Tensor,
    phi_prime: torch.Tensor,
    beta: torch.Tensor,
    n_factor: torch.Tensor,
) -> torch.Tensor:
    """N+ 整数解（可微版本使用连续近似）"""
    return torch.floor((torch.pi + beta) / (2 * torch.pi * n_factor))


def _N_minus(
    phi: torch.Tensor,
    phi_prime: torch.Tensor,
    beta: torch.Tensor,
    n_factor: torch.Tensor,
) -> torch.Tensor:
    """N- 整数解"""
    return torch.floor((torch.pi - beta) / (2 * torch.pi * n_factor))


def diffraction_loss_db(
    k: torch.Tensor,
    phi: torch.Tensor,
    phi_prime: torch.Tensor,
    beta0: torch.Tensor,
    n_factor: torch.Tensor,
    distance: torch.Tensor = 1.0,
) -> torch.Tensor:
    """绕射损耗 [dB] — 可微版本

    Args:
        k: 波数
        phi: 观察角
        phi_prime: 入射角
        beta0: Keller 锥角
        n_factor: 楔形因子
        distance: 绕射点到观察点距离 [m]

    Returns:
        loss_db: 绕射损耗 [dB]（可微分）
    """
    D = utd_diffraction_coefficient_diff(k, phi, phi_prime, beta0, n_factor)
    # 绕射场幅度: |D| / √s
    D_mag = torch.abs(D)
    diffraction_gain = D_mag / torch.sqrt(distance)
    loss_db = -20 * torch.log10(diffraction_gain + 1e-15)
    return loss_db
