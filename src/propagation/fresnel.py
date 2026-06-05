"""
Fresnel 反射与透射系数

计算平面波在介质分界面上的反射/透射,
支持 TE (s) 和 TM (p) 偏振以及完全偏振追踪。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.utils.constants import C, ETA_0


def snell_refraction_index(
    n1: complex, n2: complex, theta_i: float
) -> complex:
    """广义 Snell 定律：计算复折射角（吸收介质）

    n1 sin θ_i = n2 sin θ_t

    对于吸收介质 (n2 为复数), θ_t 是复数, 表示等幅面与等相位面不同方向。

    Args:
        n1: 入射介质折射率
        n2: 出射介质折射率（可为复数）
        theta_i: 入射角 [rad]

    Returns:
        theta_t: 复折射角 [rad]
    """
    sin_theta_i = np.sin(theta_i)
    # n1 * sin(θi) = n2 * sin(θt) → sin(θt) = n1/n2 * sin(θi)
    sin_theta_t = n1 / n2 * sin_theta_i
    return np.arcsin(sin_theta_t)


def fresnel_coefficients(
    n1: complex, n2: complex, theta_i: float
) -> Tuple[complex, complex, complex, complex]:
    """计算 Fresnel 反射/透射系数

    Args:
        n1: 入射介质折射率
        n2: 出射介质折射率
        theta_i: 入射角 [rad]

    Returns:
        (r_te, r_tm, t_te, t_mt):
            r_te: TE (s) 偏振反射系数
            r_tm: TM (p) 偏振反射系数
            t_te: TE (s) 偏振透射系数
            t_tm: TM (p) 偏振透射系数
    """
    theta_t = snell_refraction_index(n1, n2, theta_i)
    cos_theta_i = np.cos(theta_i)
    cos_theta_t = np.cos(theta_t)

    # TE (s-pol): 电场垂直于入射面
    r_te = (n1 * cos_theta_i - n2 * cos_theta_t) / \
           (n1 * cos_theta_i + n2 * cos_theta_t)
    t_te = (2 * n1 * cos_theta_i) / \
           (n1 * cos_theta_i + n2 * cos_theta_t)

    # TM (p-pol): 电场平行于入射面
    r_tm = (n2 * cos_theta_i - n1 * cos_theta_t) / \
           (n2 * cos_theta_i + n1 * cos_theta_t)
    t_tm = (2 * n1 * cos_theta_i) / \
           (n2 * cos_theta_i + n1 * cos_theta_t)

    return r_te, r_tm, t_te, t_tm


def reflectivity(r_te: complex, r_tm: complex) -> Tuple[float, float]:
    """反射率（功率反射系数）

    Returns:
        (R_te, R_tm): TE 和 TM 的功率反射率
    """
    return abs(r_te)**2, abs(r_tm)**2


def brewster_angle(n1: complex, n2: complex) -> float:
    """Brewster 角（TM 偏振零反射）

    仅当两种介质均为非吸收时有意义。
    """
    if abs(n2.imag) > 1e-10 or abs(n1.imag) > 1e-10:
        # 吸收介质中无严格 Brewster 角，返回最小值位置
        return _brewster_absorbing(n1, n2)
    return np.arctan(abs(n2 / n1))


def _brewster_absorbing(n1: complex, n2: complex) -> float:
    """吸收介质的\"Brewster 角\"——扫描找反射率最小值"""
    thetas = np.linspace(0, np.pi / 2, 1000)
    min_r, best_theta = float("inf"), 0.0
    for th in thetas:
        _, r_tm, _, _ = fresnel_coefficients(n1, n2, th)
        r = abs(r_tm)**2
        if r < min_r:
            min_r = r
            best_theta = th
    return best_theta


def refractive_index_from_permittivity(eps_r: complex) -> complex:
    """复介电常数 → 复折射率

    n = √ε_r
    """
    return np.sqrt(eps_r)
