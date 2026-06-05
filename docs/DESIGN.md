# 高频电磁波自由空间传播仿真框架

## 研究背景

### 频段范围：10 GHz – 1 THz

| 频段 | 波长 | 典型应用 |
|------|------|----------|
| 10–30 GHz (X/Ku/Ka) | 3–1 cm | 卫星通信, 5G NR |
| 30–100 GHz (V/W) | 10–3 mm | 毫米波雷达, 6G 候选 |
| 100–300 GHz (sub-THz) | 3–1 mm | 太赫兹成像, 高速通信 |
| 0.3–1 THz | 1–0.3 mm | 安检成像, 光谱学 |

### 物理模型的选择

在此频段，波长远小于典型场景尺度（λ ≪ 场景尺寸），因此**几何光学 (Geometric Optics, GO)** 近似成立。在此基础上：

- **反射/折射** → Snell 定律 + Fresnel 方程（偏振依赖）
- **绕射** → 一致性绕射理论 (UTD) / 几何绕射理论 (GTD)
- **传播损耗** → Friis 传输方程 + 大气吸收（ITU-R P.676）
- **材料响应** → 复介电常数 → 反射率/透射率

### 与 Mitsuba 的关联

Mitsuba 3 是基于物理的光线追踪渲染器。本项目借鉴其核心思想：

```
Mitsuba (可见光)        本项目 (10GHz–1THz)
────────────────────────────────────────
BSDF / 表面散射          Fresnel 反射/透射
介质/导体材料            复介电常数材料
路径追踪                射线发射 (SBR)
双向散射分布函数 (BSDF)   双站雷达截面 (BSC)
重要性采样              自适应射线细分
偏振                    完全偏振追踪
```

---

## 研究创新点

### 1. 宽频带材料的统一电磁参数模型

**问题**: 10 GHz – 1 THz 跨越 2 个量级，材料的电磁参数变化剧烈（尤其是水、混凝土、织物）。

**创新**: 构建基于**复介电常数数据库 + 多极点 Debye 模型**的宽谱材料模型，覆盖建筑材料、人体组织、典型室内材料。

### 2. 混合 GO/UTD 一致性绕射引擎

**问题**: 传统光追假设表面足够大，忽略边缘绕射。在毫米波频段，边缘绕射是 NLOS (非视距) 传播的主导机制。

**创新**: 实现**一致性绕射理论 (UTD)**，支持：
- 直边绕射 (Keller 锥)
- 多次绕射
- 绕射 + 反射耦合路径

### 3. 物理感知的自适应射线细分 (Adaptive Ray Marching)

**问题**: 均匀射线发射浪费计算资源，关键区域（边缘、多径）需要更高采样密度。

**创新**: 基于场景复杂度和路径增益的自适应射线细分：
- 首次反弹后根据 Fresnel 系数的变化率决定细分
- 边缘检测后自动在 Keller 锥方向加密射线

### 4. 偏振全链路追踪

**问题**: 传统无线信道模型忽略偏振。在 THz 频段，偏振是关键的调制维度。

**创新**: 完整追踪 **Stokes 矢量** 沿传播路径的变化：
- 反射/透射 → Mueller 矩阵运算
- 绕射 → 边缘绕射偏振修正
- 输出 → 极化态变化 + 交叉极化鉴别度 (XPD)

### 5. 可微传播模型 (Differentiable Propagation)

**问题**: 传统仿真是前向的，无法反向传播梯度用于逆设计。

**创新**: 基于 PyTorch 的可微射线追踪：
- 场景参数（材料、位置） → 梯度 → 优化
- 支持信道估计的超分辨率重建
- 可用于智能超表面 (RIS) 的逆设计

---

## 算法设计

### 核心算法：射线发射法 (SBR)

```
for each 发射射线:
    1. 场景求交 (BVH 加速)
    2. 如果是首次碰撞:
       a. 计算 Fresnel 反射/透射系数
       b. 如果反射系数 > threshold → 分裂为反射射线
       c. 如果透射系数 > threshold → 分裂为透射射线
       d. 如果是边缘 → 生成 Keller 锥绕射射线
    3. 如果是二次及以上碰撞:
       a. 同 2a–2d
       b. 如果达到最大弹射次数 → 终止
    4. 如果射线到达接收球 → 记录路径 + 场强
```

### UTD 绕射系数

一致性绕射系数（Kouyoumjian-Pathak 形式）：

```
D_s,h(φ, φ', β0) = 
  -exp(-jπ/4) / (2n√(2πk) sin(β0)) ×
  [cot((π + (φ-φ'))/(2n)) · F(kL a⁺(φ-φ'))
   + cot((π - (φ-φ'))/(2n)) · F(kL a⁻(φ-φ'))
   + R0_s,h · cot((π - (φ+φ'))/(2n)) · F(kL a⁻(φ+φ'))
   + Rn_s,h · cot((π + (φ+φ'))/(2n)) · F(kL a⁺(φ+φ'))]
```

其中 F(x) 是 Fresnel 过渡函数，R 是反射系数。

### Fresnel 反射系数（TE/TM 偏振）

```
r_TE = (n1 cos θi - n2 cos θt) / (n1 cos θi + n2 cos θt)
r_TM = (n2 cos θi - n1 cos θt) / (n2 cos θi + n1 cos θt)
```

其中 n = √ε_r 是复折射率，θ_i 和 θ_t 通过广义 Snell 定律关联。

### 路径损耗

```
PL(f, d) = 20 log10(4π d f / c) + A_atm(f, d) + L_ref(f, θ) + L_diff(f, θ)
```

- 第一项：Friis 自由空间损耗
- 第二项：ITU-R P.676 大气吸收
- 第三项：Fresnel 反射损耗
- 第四项：UTD 绕射损耗

---

## 框架架构

```
em-wave-propagation/
├── docs/                          # 文档
│   ├── design.md                  # 本设计文档
│   └── references/                # 参考文献
│
├── src/
│   ├── core/
│   │   ├── ray.py                 # Ray 数据结构
│   │   ├── intersection.py        # 交点数据
│   │   ├── spectrum.py            # 频谱采样
│   │   └── polarization.py        # Stokes/Mueller 偏振
│   │
│   ├── geometry/
│   │   ├── scene.py               # 场景图
│   │   ├── primitive.py           # 基元 (三角面/球/圆柱)
│   │   ├── bvh.py                 # BVH 加速结构
│   │   └── edge.py                # 边缘检测 + Keller 锥
│   │
│   ├── materials/
│   │   ├── base.py                # Material base (复折射率)
│   │   ├── debye.py               # Debye 多极点模型
│   │   ├── itu.py                 # ITU-R 大气吸收模型
│   │   └── database.py            # 材料数据库查询
│   │
│   ├── propagation/
│   │   ├── fresnel.py             # Fresnel 反射/透射
│   │   ├── utd.py                 # UTD 绕射系数
│   │   ├── sbr.py                 # 射线发射引擎
│   │   ├── path_tracer.py         # 路径追踪 (Mitsuba 风格)
│   │   └── loss.py                # 路径损耗计算
│   │
│   ├── renderer/
│   │   ├── field.py               # 场强渲染 (功率延迟分布)
│   │   ├── coverage.py            # 覆盖图渲染
│   │   └── polarimetry.py         # 极化渲染
│   │
│   └── utils/
│       ├── constants.py           # 物理常数
│       ├── units.py               # 单位转换
│       └── io.py                  # 场景导入/导出
│
├── config/
│   ├── materials.json             # 材料数据库
│   └── scenarios/                 # 场景配置
│       ├── indoor_office.yaml
│       └── urban_canyon.yaml
│
├── examples/
│   ├── simple_reflection.py
│   ├── edge_diffraction.py
│   └── coverage_map.py
│
├── tests/
│   ├── test_fresnel.py
│   ├── test_utd.py
│   └── test_bvh.py
│
└── requirements.txt
```

---

## MIT 许可证
