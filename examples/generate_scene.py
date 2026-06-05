"""
生成示例 3D 室内场景 (OBJ 格式)

创建一个含隔墙、窗户和金属柜的 10m×8m×3m 房间。
"""
from __future__ import annotations

import os
import numpy as np


def generate_office_scene(output_dir: str = ".") -> str:
    """生成办公场景 OBJ + MTL 文件

    Returns:
        主 .obj 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, "indoor_office.obj")
    mtl_path = os.path.join(output_dir, "indoor_office.mtl")

    # ── MTL 材质文件 ───────────────────────────────────────────
    with open(mtl_path, "w") as f:
        f.write("# Indoor Office Scene - Material Definitions\n")
        f.write("newmtl Wall\n")
        f.write("Kd 0.65 0.60 0.55\n")  # 米色
        f.write("newmtl Floor\n")
        f.write("Kd 0.40 0.35 0.30\n")
        f.write("newmtl Ceiling\n")
        f.write("Kd 0.85 0.82 0.80\n")
        f.write("newmtl Glass\n")
        f.write("Kd 0.53 0.81 0.92\n")
        f.write("newmtl Metal\n")
        f.write("Kd 0.75 0.75 0.75\n")

    # ── OBJ 文件 ───────────────────────────────────────────────
    with open(obj_path, "w") as f:
        f.write("# Indoor Office Scene - EM Wave Propagation Demo\n")
        f.write(f"mtllib {os.path.basename(mtl_path)}\n\n")

        vert_idx = 1

        def add_box(name, center, size, material, f):
            """添加长方体到 OBJ"""
            nonlocal vert_idx
            x, y, z = center
            dx, dy, dz = size[0]/2, size[1]/2, size[2]/2

            # 8 个顶点
            verts = [
                (x-dx, y-dy, z-dz), (x+dx, y-dy, z-dz),
                (x+dx, y+dy, z-dz), (x-dx, y+dy, z-dz),
                (x-dx, y-dy, z+dz), (x+dx, y-dy, z+dz),
                (x+dx, y+dy, z+dz), (x-dx, y+dy, z+dz),
            ]
            for v in verts:
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")

            # 6 个面 (每个面 2 个三角形)
            faces = [
                (0, 1, 2), (0, 2, 3),  # bottom
                (4, 6, 5), (4, 7, 6),  # top
                (0, 4, 5), (0, 5, 1),  # front
                (2, 6, 7), (2, 7, 3),  # back
                (0, 3, 7), (0, 7, 4),  # left
                (1, 5, 6), (1, 6, 2),  # right
            ]

            f.write(f"usemtl {material}\n")
            f.write(f"o {name}\n")
            for a, b, c in faces:
                f.write(f"f {a+vert_idx} {b+vert_idx} {c+vert_idx}\n")

            vert_idx += 8

        # 房间
        add_box("Floor", [5.0, -0.1, 1.5], [10.0, 0.2, 3.0], "Floor", f)
        add_box("Ceiling", [5.0, 3.1, 1.5], [10.0, 0.2, 3.0], "Ceiling", f)
        add_box("BackWall", [-0.1, 1.5, 1.5], [0.2, 3.0, 3.0], "Wall", f)
        add_box("FrontWall", [10.1, 1.5, 1.5], [0.2, 3.0, 3.0], "Wall", f)
        add_box("LeftWall", [5.0, 1.5, -0.1], [10.0, 3.0, 0.2], "Wall", f)
        add_box("RightWall", [5.0, 1.5, 3.1], [10.0, 3.0, 0.2], "Wall", f)

        # 隔墙（中间，带玻璃窗区域用 Metal 标记）
        add_box("Partition", [5.0, 1.5, 1.5], [0.15, 3.0, 2.8], "Wall", f)
        # 隔墙上的玻璃窗
        add_box("Window", [5.0, 1.8, 1.8], [0.16, 0.8, 1.2], "Glass", f)

        # 金属柜
        add_box("Cabinet", [2.0, 0.5, 2.8], [0.8, 1.0, 0.6], "Metal", f)

        print(f"  ✅ 已生成: {obj_path}")
        print(f"     {vert_idx - 1} 个顶点")

    return obj_path


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "../examples/models")
    generate_office_scene(out_dir)
