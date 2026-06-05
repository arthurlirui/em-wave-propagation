"""
一致性绕射理论 (UTD) — 边缘绕射系数

实现 Kouyoumjian-Pathak 形式的一致性绕射系数，
支持直边（楔形）绕射，含 Fresnel 过渡函数。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import erf

from src.utils.constants import C, wavelength, wavenumber


def _fresnel_transition(x: float) -> complex:
    """Fresnel 过渡函数 F(x)

    F(x) = 2j√x · exp(jx) · ∫_{√x}^{∞} exp(-jτ²) dτ

    使用辅助函数近似计算。
    F(x) 在 x → 0 时趋近 0，在 x → ∞ 时趋近 1。

    Args:
        x: 过渡参数

    Returns:
        F(x): 复数 Fresnel 过渡函数
    """
    x = max(x, 1e-10)  # 避免数值问题

    # 小参量展开
    if x < 0.1:
        sqrt_x = np.sqrt(x)
        F = (np.sqrt(np.pi * x) - 2j * x * np.exp(1j * x)
             - (2 / 3) * (1j * x)**1.5) * np.exp(-1j * (x + np.pi / 4))
        return F

    # 使用复数 Fresnel 积分的渐进展开
    sqrt_x = np.sqrt(x)
    # ∫_{√x}^{∞} exp(-jτ²) dτ = √(π/4) * (1 - C(√x) - jS(√x))
    # 其中 C, S 是 Cornu 螺线

    from scipy.special import fresnel as scipy_fresnel
    S, C = scipy_fresnel(sqrt_x)
    integral = (np.sqrt(np.pi / 8)) * (1 - 2*C - 1j * (1 - 2*S))

    F = 2j * sqrt_x * np.exp(1j * x) * integral
    return F


def _cot(x: float) -> complex:
    """安全的余切函数，避免极点"""
    if abs(x) < 1e-10:
        x = 1e-10
    return 1.0 / np.tan(x)


def utd_diffraction_coefficient(
    k: float,
    phi: float,
    phi_prime: float,
    beta0: float,
    n: float,
    r0: float = 0.0,
    rn: float = 0.0,
) -> complex:
    """计算 UTD 绕射系数 D_s 和 D_h（软/硬边界条件）

    基于 Kouyoumjian-Pathak 的经典形式。

    Args:
        k: 波数 [rad/m]
        phi: 观察角 [rad]（绕射射线与 0-face 夹角）
        phi_prime: 入射角 [rad]（入射射线与 0-face 夹角）
        beta0: 绕射锥半角 [rad]（射线与边缘夹角）
        n: 楔形因子 (nπ = 内角), n=2 为半平面
        r0: 0-face 反射系数
        rn: n-face 反射系数

    Returns:
        D: 标量绕射系数（硬边界 H-polarization）
        对于软边界 (E-polarization) 需取负号
    """
    if beta0 < 1e-10:
        return 0.0

    L = _calculate_distance_factor(k, phi, phi_prime, beta0)

    # Fresnel 过渡函数的参数
    def a_plus(beta: float) -> float:
        return 2 * np.cos((2 * np.pi * n * _N_plus(phi, phi_prime, beta, n) - (phi - phi_prime)) / 2)**2

    def a_minus(beta: float) -> float:
        return 2 * np.cos((2 * np.pi * n * _N_minus(phi, phi_prime, beta, n) - (phi - phi_prime)) / 2)**2

    # 四项求和
    term1 = _cot((np.pi + (phi - phi_prime)) / (2 * n)) * _fresnel_transition(k * L * a_plus(phi - phi_prime))
    term2 = _cot((np.pi - (phi - phi_prime)) / (2 * n)) * _fresnel_transition(k * L * a_minus(phi - phi_prime))
    term3 = r0 * _cot((np.pi - (phi + phi_prime)) / (2 * n)) * _fresnel_transition(k * L * a_minus(phi + phi_prime))
    term4 = rn * _cot((np.pi + (phi + phi_prime)) / (2 * n)) * _fresnel_transition(k * L * a_plus(phi + phi_prime))

    D = (-np.exp(-1j * np.pi / 4)
         / (2 * n * np.sqrt(2 * np.pi * k) * np.sin(beta0))
         * (term1 + term2 + term3 + term4))

    return D


def _N_plus(phi: float, phi_prime: float, beta: float, wedge_n: float) -> int:
    """N+ 整数解"""
    return np.floor((np.pi + beta) / (2 * np.pi * wedge_n))


def _N_minus(phi: float, phi_prime: float, beta: float, wedge_n: float) -> int:
    """N- 整数解"""
    return np.floor((np.pi - beta) / (2 * np.pi * wedge_n))


def _calculate_distance_factor(
    k: float, phi: float, phi_prime: float, beta0: float
) -> float:
    """计算距离因子 L（用于 Fresnel 过渡函数）

    对于平面波入射, L = s·sin²(β₀)
    其中 s 是从绕射点到观察点的距离。
    """
    # 默认假设单位距离；实际应用中需传入 s
    s = 1.0
    return s * np.sin(beta0)**2


def diffracted_field(
    e_incident: complex,
    k: float,
    s: float,
    phi: float,
    phi_prime: float,
    beta0: float,
    n: float,
    polarization: str = "hard",
) -> complex:
    """计算绕射场

    E_diff = E_inc · D · exp(-jks) / √s

    Args:
        e_incident: 入射场幅度
        k: 波数
        s: 绕射点到观察点距离
        phi: 观察角
        phi_prime: 入射角
        beta0: 绕射锥角
        n: 楔形因子
        polarization: "hard" (H-pol) 或 "soft" (E-pol)

    Returns:
        绕射场复数幅度
    """
    D = utd_diffraction_coefficient(k, phi, phi_prime, beta0, n)

    # 软边界条件取负号
    if polarization == "soft":
        D = -D

    return e_incident * D * np.exp(-1j * k * s) / np.sqrt(s)
