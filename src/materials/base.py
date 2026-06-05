"""
材料电磁参数模型

提供复介电常数 ε_r(f) 的频率相关模型：
- Debye 多极点模型
- ITU-R 大气吸收模型
- 常见建筑材料数据库
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DebyePole:
    """Debye 极点"""
    tau: float       # 弛豫时间 [s]
    delta_eps: float # 介电强度
    alpha: float = 0.0  # Cole-Cole 分布参数


@dataclass
class Material:
    """电磁材料

    使用多极点 Debye 模型描述复介电常数的频率响应。
    """
    name: str
    eps_inf: float              # 高频介电常数 (ε∞)
    poles: List[DebyePole]      # Debye 极点列表
    conductivity: float = 0.0   # 直流电导率 [S/m]
    thickness: Optional[float] = None  # 材料厚度 [m]

    def permittivity(self, f: float) -> complex:
        """计算给定频率的复介电常数

        ε_r(f) = ε_∞ + Σ Δε_k / (1 + (jωτ_k)^(1-α_k)) + σ / (jωε_0)

        Args:
            f: 频率 [Hz]

        Returns:
            复相对介电常数 ε_r = ε' - jε''
        """
        omega = 2 * np.pi * f
        eps = complex(self.eps_inf, 0.0)

        for pole in self.poles:
            tau_jw = 1j * omega * pole.tau
            if pole.alpha > 0:
                # Cole-Cole 模型
                denominator = 1.0 + (tau_jw) ** (1 - pole.alpha)
            else:
                # 标准 Debye
                denominator = 1.0 + tau_jw
            eps += pole.delta_eps / denominator

        # 直流电导率贡献
        if self.conductivity > 0:
            eps -= 1j * self.conductivity / (omega * 8.854187817e-12)

        return eps

    def refractive_index(self, f: float) -> complex:
        """计算复折射率 n = √ε_r"""
        return np.sqrt(self.permittivity(f))

    def reflection_loss_db(self, f: float, theta: float = 0.0) -> float:
        """给定频率和入射角的反射损耗 [dB]"""
        n = self.refractive_index(f)
        from src.propagation.fresnel import fresnel_coefficients, reflectivity
        r_te, r_tm, _, _ = fresnel_coefficients(1.0, n, theta)
        r_te_pow, r_tm_pow = reflectivity(r_te, r_tm)
        # 平均反射损耗
        avg_r = (r_te_pow + r_tm_pow) / 2
        return -10 * np.log10(max(avg_r, 1e-15))


# ── 材料数据库 ─────────────────────────────────────────────────────

# 参考来源: ITU-R P.2040, FCC mmWave 材料测量

MATERIAL_DATABASE: Dict[str, Material] = {
    "vacuum": Material(
        name="Vacuum",
        eps_inf=1.0,
        poles=[],
    ),

    "drywall": Material(
        name="Drywall (Gypsum)",
        eps_inf=2.8,
        poles=[
            DebyePole(tau=1e-12, delta_eps=0.5),
        ],
        conductivity=0.01,
    ),

    "concrete": Material(
        name="Concrete",
        eps_inf=4.5,
        poles=[
            DebyePole(tau=5e-11, delta_eps=1.5),
            DebyePole(tau=1e-12, delta_eps=0.8),
        ],
        conductivity=0.05,
    ),

    "glass": Material(
        name="Window Glass",
        eps_inf=4.0,
        poles=[
            DebyePole(tau=2e-12, delta_eps=1.5),
        ],
        conductivity=0.001,
    ),

    "wood": Material(
        name="Wood",
        eps_inf=1.8,
        poles=[
            DebyePole(tau=5e-11, delta_eps=0.5),
        ],
        conductivity=0.005,
    ),

    "human_skin": Material(
        name="Human Skin (dry)",
        eps_inf=4.0,
        poles=[
            DebyePole(tau=1e-11, delta_eps=6.0),
            DebyePole(tau=2e-13, delta_eps=20.0),
        ],
        conductivity=0.5,
    ),

    "metal": Material(
        name="Ideal Conductor (PEC)",
        eps_inf=1.0,
        poles=[],
        conductivity=1e10,
    ),
}


def get_material(name: str) -> Material:
    """按名称查询材料"""
    if name.lower() in MATERIAL_DATABASE:
        return MATERIAL_DATABASE[name.lower()]
    raise KeyError(f"Unknown material: {name}. Available: {list(MATERIAL_DATABASE.keys())}")


def list_materials() -> List[str]:
    """列出所有可用材料"""
    return list(MATERIAL_DATABASE.keys())
