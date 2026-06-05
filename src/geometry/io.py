"""
3D 模型导入器

支持格式：
- OBJ (Wavefront) — 三角网格 + 材质引用
- STL (二进制/ASCII) — 三角网格
- 自定义 JSON — 场景描述 + 电磁参数

材质通过 .mtl 文件或外部映射文件赋予电磁参数。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.geometry.primitive import SceneObject, Triangle, Edge


@dataclass
class ObjMesh:
    """OBJ 网格数据"""
    vertices: List[np.ndarray] = field(default_factory=list)
    normals: List[np.ndarray] = field(default_factory=list)
    faces: List[Tuple] = field(default_factory=list)  # (v_idx, vt_idx, vn_idx)
    material_name: str = "vacuum"


def load_obj(filepath: str, material_map: Optional[Dict[str, str]] = None) -> List[SceneObject]:
    """加载 OBJ 文件为 SceneObject 列表

    Args:
        filepath: .obj 文件路径
        material_map: OBJ 材质名 → 电磁材料名 映射
                      例如 {"Wall": "concrete", "Window": "glass"}

    Returns:
        SceneObject 列表（按材质分组）
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"OBJ file not found: {filepath}")

    material_map = material_map or {}

    vertices: List[np.ndarray] = [np.array([0, 0, 0])]  # 1-indexed
    normals: List[np.ndarray] = [np.array([0, 0, 0])]
    texcoords: List[np.ndarray] = [np.array([0, 0])]

    # 按材质分组的三角面
    mesh_groups: Dict[str, ObjMesh] = {}
    current_material = "vacuum"

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            cmd = parts[0]

            if cmd == "v":
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append(np.array([x, y, z]))

            elif cmd == "vn":
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                n = np.array([x, y, z])
                norm = np.linalg.norm(n)
                if norm > 0:
                    n /= norm
                normals.append(n)

            elif cmd == "vt":
                u, v = float(parts[1]), float(parts[2])
                texcoords.append(np.array([u, v]))

            elif cmd == "usemtl":
                current_material = parts[1]
                if current_material not in mesh_groups:
                    mesh_groups[current_material] = ObjMesh(material_name=current_material)

            elif cmd == "f":
                # 解析面定义: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                face_verts = []
                face_normals = []
                for p in parts[1:4]:  # 只取前三个（三角形）
                    indices = p.split("/")
                    vi = int(indices[0])
                    face_verts.append(vertices[vi])
                    if len(indices) >= 3 and indices[2]:
                        ni = int(indices[2])
                        face_normals.append(normals[ni])

                if current_material not in mesh_groups:
                    mesh_groups[current_material] = ObjMesh(material_name=current_material)

                mesh = mesh_groups[current_material]
                avg_normal = np.mean(face_normals, axis=0) if face_normals else None
                if avg_normal is not None and np.linalg.norm(avg_normal) > 0:
                    avg_normal /= np.linalg.norm(avg_normal)

                mesh.faces.append((face_verts, avg_normal))

    # 按材质分组构建 SceneObject
    objects = []
    for mat_name, mesh in mesh_groups.items():
        em_material = material_map.get(mat_name, mat_name)
        obj = SceneObject(name=mat_name)

        for fv, fn in mesh.faces:
            if len(fv) == 3:
                tri = Triangle(
                    vertices=np.array(fv, dtype=np.float64),
                    material=em_material,
                )
                if fn is not None:
                    tri.normal = fn
                obj.triangles.append(tri)

        if obj.triangles:
            objects.append(obj)

    return objects


def load_stl(filepath: str, material: str = "metal") -> SceneObject:
    """加载 STL 文件（二进制或 ASCII）为 SceneObject

    Args:
        filepath: .stl 文件路径
        material: 赋予所有三角面的电磁材料

    Returns:
        包含所有三角面的 SceneObject
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"STL file not found: {filepath}")

    obj = SceneObject(name=os.path.basename(filepath))

    with open(filepath, "rb") as f:
        header = f.read(80).decode("ascii", errors="ignore").strip()

    if header.startswith("solid"):
        _load_stl_ascii(filepath, obj, material)
    else:
        _load_stl_binary(filepath, obj, material)

    return obj


def _load_stl_ascii(filepath: str, obj: SceneObject, material: str):
    """加载 ASCII STL"""
    import re
    with open(filepath, "r") as f:
        content = f.read()

    # 按 facet 分割
    facet_pattern = re.compile(
        r"facet normal\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*"
        r"outer loop\s*"
        r"vertex\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*"
        r"vertex\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*"
        r"vertex\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*"
        r"endloop\s*"
        r"endfacet",
        re.DOTALL,
    )

    for match in facet_pattern.finditer(content):
        vals = [float(g) for g in match.groups()]
        normal = np.array(vals[0:3])
        verts = np.array([
            [vals[3], vals[4], vals[5]],
            [vals[6], vals[7], vals[8]],
            [vals[9], vals[10], vals[11]],
        ], dtype=np.float64)
        tri = Triangle(vertices=verts, material=material)
        tri.normal = normal
        obj.triangles.append(tri)


def _load_stl_binary(filepath: str, obj: SceneObject, material: str):
    """加载二进制 STL"""
    import struct
    with open(filepath, "rb") as f:
        f.read(80)  # header
        num_triangles = struct.unpack("<I", f.read(4))[0]

        for _ in range(num_triangles):
            data = f.read(50)  # 12+36+2 = 50 bytes per triangle
            if len(data) < 50:
                break
            vals = struct.unpack("<12fH", data)
            normal = np.array(vals[0:3])
            verts = np.array([
                [vals[3], vals[4], vals[5]],
                [vals[6], vals[7], vals[8]],
                [vals[9], vals[10], vals[11]],
            ], dtype=np.float64)
            tri = Triangle(vertices=verts, material=material)
            tri.normal = normal
            obj.triangles.append(tri)


def load_scene_json(filepath: str) -> List[SceneObject]:
    """加载 JSON 场景描述文件

    JSON 格式:
    {
        "objects": [
            {
                "name": "room",
                "type": "box",
                "center": [5, 1.5, 1.5],
                "size": [10, 3, 3],
                "material": "drywall"
            }
        ],
        "sources": [...],
        "receivers": [...]
    }
    """
    from src.core.ray import Source, Receiver

    with open(filepath, "r") as f:
        data = json.load(f)

    objects = []
    for obj_data in data.get("objects", []):
        if obj_data.get("type") == "box":
            obj = SceneObject(name=obj_data["name"])
            obj.add_box(
                center=obj_data["center"],
                size=obj_data["size"],
                material=obj_data.get("material", "concrete"),
            )
            objects.append(obj)

        elif obj_data.get("type") == "mesh":
            mesh_path = obj_data.get("file")
            if mesh_path and os.path.isfile(mesh_path):
                mat_map = obj_data.get("material_map", {})
                loaded = load_obj(mesh_path, mat_map)
                objects.extend(loaded)

    return objects
