"""
可微 Fresnel 反射/透射 — PyTorch Autograd

所有计算使用 PyTorch 张量，支持对 n₁, n₂, θ 求导。
用于逆设计与参数优化。
"""

from __future__ import annotations

from typing import Tuple

import torch


def fresnel_reflection_coefficients(
    n1: torch.Tensor,
    n2: torch.Tensor,
    theta_i: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """可微 Fresnel 反射系数

    Args:
        n1: 入射介质复折射率 (...,) — 可微分
        n2: 出射介质复折射率 (...,) — 可微分（最常用的优化变量）
        theta_i: 入射角 [rad] (...,) — 可微分

    Returns:
        r_te: TE (s) 偏振反射系数 (..., complex)
        r_tm: TM (p) 偏振反射系数 (..., complex)
    """
    # 广义 Snell 定律
    sin_theta_i = torch.sin(theta_i)
    sin_theta_t = (n1 / n2) * sin_theta_i

    # 对复数取 arcsin，处理分支切割
    cos_theta_t = torch.sqrt(1.0 - sin_theta_t**2)

    cos_theta_i = torch.cos(theta_i)

    # TE (s-pol): r = (n₁cosθᵢ - n₂cosθₜ) / (n₁cosθᵢ + n₂cosθₜ)
    denom_te = n1 * cos_theta_i + n2 * cos_theta_t
    r_te = (n1 * cos_theta_i - n2 * cos_theta_t) / denom_te

    # TM (p-pol): r = (n₂cosθᵢ - n₁cosθₜ) / (n₂cosθᵢ + n₁cosθₜ)
    denom_tm = n2 * cos_theta_i + n1 * cos_theta_t
    r_tm = (n2 * cos_theta_i - n1 * cos_theta_t) / denom_tm

    return r_te, r_tm


def fresnel_reflectance(
    n1: torch.Tensor,
    n2: torch.Tensor,
    theta_i: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """可微功率反射率（Fresnel 反射率的平方模）

    Returns:
        R_te: TE 功率反射率 [0,1] — 可微分
        R_tm: TM 功率反射率 [0,1] — 可微分
    """
    r_te, r_tm = fresnel_reflection_coefficients(n1, n2, theta_i)
    R_te = torch.abs(r_te) ** 2
    R_tm = torch.abs(r_tm) ** 2
    return R_te, R_tm


def fresnel_transmission_coefficients(
    n1: torch.Tensor,
    n2: torch.Tensor,
    theta_i: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """可微 Fresnel 透射系数

    Returns:
        t_te: TE 透射系数 (complex)
        t_tm: TM 透射系数 (complex)
    """
    sin_theta_i = torch.sin(theta_i)
    sin_theta_t = (n1 / n2) * sin_theta_i
    cos_theta_i = torch.cos(theta_i)
    cos_theta_t = torch.sqrt(1.0 - sin_theta_t**2)

    denom_te = n1 * cos_theta_i + n2 * cos_theta_t
    t_te = (2.0 * n1 * cos_theta_i) / denom_te

    denom_tm = n2 * cos_theta_i + n1 * cos_theta_t
    t_tm = (2.0 * n1 * cos_theta_i) / denom_tm

    return t_te, t_tm


def path_loss_fresnel(
    n1: torch.Tensor,
    n2: torch.Tensor,
    theta_i: torch.Tensor,
    num_reflections: int = 1,
) -> torch.Tensor:
    """多次反射后的累积 Fresnel 损耗 [dB]

    可微分版本，支持对 n₂ 求导，用于材料参数优化。

    Args:
        n1: 入射介质折射率
        n2: 反射介质折射率（优化目标）
        theta_i: 入射角
        num_reflections: 反射次数

    Returns:
        loss_db: 累积反射损耗 [dB] — 可微分
    """
    R_te, R_tm = fresnel_reflectance(n1, n2, theta_i)
    # 平均 TE/TM 反射率
    R_avg = (R_te + R_tm) / 2.0
    # 多次反射
    R_total = R_avg ** num_reflections
    # dB 转换（加极小值避免 log(0)）
    loss_db = -10.0 * torch.log10(R_total + 1e-15)
    return loss_db


def brewster_angle_approx(
    n1: torch.Tensor,
    n2: torch.Tensor,
) -> torch.Tensor:
    """近似 Brewster 角（适用于弱吸收介质）

    通过扫描求 TM 反射率最小值的位置，使用 Soft argmin 保持可微性。

    Args:
        n1: 入射介质折射率
        n2: 出射介质折射率

    Returns:
        theta_b: Brewster 角 [rad]
    """
    # 在 [0, π/2] 上采样
    thetas = torch.linspace(0.0, torch.pi / 2, 200, dtype=n1.dtype)
    _, R_tm_all = fresnel_reflectance(
        n1.expand_as(thetas),
        n2.expand_as(thetas),
        thetas,
    )
    # Soft argmin（可微的 argmin）
    weights = torch.softmax(-R_tm_all.real / 0.01, dim=0)
    theta_b = torch.sum(thetas * weights)
    return theta_b
