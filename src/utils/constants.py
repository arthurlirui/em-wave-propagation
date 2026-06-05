"""
物理常数与单位转换
"""

from __future__ import annotations

import numpy as np

# ── 基本物理常数 ────────────────────────────────────────────────────
C = 299_792_458  # 光速 [m/s]
EPSILON_0 = 8.854187817e-12  # 真空介电常数 [F/m]
MU_0 = 1.25663706212e-6     # 真空磁导率 [H/m]
ETA_0 = np.sqrt(MU_0 / EPSILON_0)  # 真空波阻抗 ≈ 377 Ω

# ── 频段定义 ────────────────────────────────────────────────────────
F_MIN = 10e9    # 最低频率 10 GHz
F_MAX = 1e12    # 最高频率 1 THz


def wavelength(f: float) -> float:
    """频率 → 波长 [m]"""
    return C / f


def wavenumber(f: float) -> float:
    """频率 → 波数 [rad/m]"""
    return 2 * np.pi * f / C


def frequency_from_wavelength(lam: float) -> float:
    """波长 → 频率 [Hz]"""
    return C / lam


def to_db(value: float) -> float:
    """线性值 → dB"""
    return 20 * np.log10(max(value, 1e-15))
