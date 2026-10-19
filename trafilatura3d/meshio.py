"""Meshes for the 3D View and the bakers: loading 3D scene files (OBJ, glTF/GLB, PLY, STL, OFF through
trimesh; USD/USDA/USDC/USDZ through Pixar's usd-core; FBX through ufbx, see fbx.py) into renderer-ready
arrays — normals and UVs filled in when missing, tangents from the UVs with the bitangent's handedness,
consistent winding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MESH_SUFFIXES = (".fbx", ".obj", ".gltf", ".glb", ".ply", ".stl", ".off", ".usd", ".usda", ".usdc", ".usdz")
MESH_FILTER = f"3D scenes ({' '.join('*' + s for s in MESH_SUFFIXES)})"


@dataclass
class Submesh:
    start: int  # first index
    count: int
    material: int


@dataclass
class MeshObject:
    name: str
    pos: np.ndarray
    nrm: np.ndarray
    tan: np.ndarray  # xyz + handedness
    uv: np.ndarray
    idx: np.ndarray
    submeshes: list
    visible: bool = True
    colors: np.ndarray | None = None  # per vertex RGBA (float) when the file has vertex colors
    material_colors: list | None = None  # RGBA per material of the file (its base / diffuse color)

    @property
    def triangles(self):
        return self.idx.reshape(-1, 3)

    def bounds(self):
        return self.pos.min(0), self.pos.max(0)



def wind(pos, nrm, idx):
    """Triangles counter-clockwise around their normals (the front face): the shader flips the
    normal of back faces, so a clockwise triangle would be shaded from inside."""
    p0, p1, p2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    n = nrm[idx[:, 0]] + nrm[idx[:, 1]] + nrm[idx[:, 2]]
    back = np.sum(np.cross(p1 - p0, p2 - p0) * n, -1) < 0
    idx = idx.copy()
    idx[back] = idx[back][:, [0, 2, 1]]
    return idx


def handedness(pos, nrm, tan, uv, idx):
    """Per vertex +1 when +V runs along cross(normal, tangent), -1 otherwise (from the triangles'
    UV derivatives), so the shader rebuilds the bitangent the way the UVs actually go."""
    p0, p1, p2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    u0, u1, u2 = uv[idx[:, 0]], uv[idx[:, 1]], uv[idx[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, u1 - u0, u2 - u0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    dpdv = (e2 * d1[:, :1] - e1 * d2[:, :1]) / np.where(np.abs(det) > 1e-12, det, 1.0)[:, None]
    acc = np.zeros(len(pos))
    for k in range(3):
        v = idx[:, k]
        np.add.at(acc, v, np.sum(np.cross(nrm[v], tan[v]) * dpdv, -1))
    return np.where(acc < 0, -1.0, 1.0).astype(np.float32)



def finish(name, pos, nrm, uv, faces, groups, n_materials):
    """Arrays for the renderer: normals and UVs filled in when missing, tangents from the UVs,
    consistent winding, scaled to fit the primitives' size (about 1.6 across)."""
    pos = np.asarray(pos, np.float64)
    faces = np.asarray(faces, np.int64).reshape(-1, 3)
    if nrm is None or len(nrm) != len(pos):
        nrm = vertex_normals(pos, faces)
    nrm = np.asarray(nrm, np.float64)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    if uv is None or len(uv) != len(pos):
        uv = box_uv(pos, nrm)
    uv = np.asarray(uv, np.float64)
    tan = vertex_tangents(pos, nrm, uv, faces)
    idx = wind(pos, nrm, faces.astype(np.uint32))
    hand = handedness(pos.astype(np.float32), nrm.astype(np.float32), tan.astype(np.float32),
                              uv.astype(np.float32), idx)
    order = np.argsort(groups, kind="stable") if groups is not None else np.arange(len(idx))
    idx = idx[order]
    groups = np.asarray(groups)[order] if groups is not None else np.zeros(len(idx), int)
    subs = []
    for m in np.unique(groups):
        rows = np.nonzero(groups == m)[0]
        subs.append(Submesh(int(rows[0]) * 3, len(rows) * 3, int(min(m, n_materials - 1))))
    return MeshObject(name, pos.astype(np.float32), nrm.astype(np.float32),
                      np.concatenate([tan, hand[:, None]], 1).astype(np.float32), uv.astype(np.float32),
                      idx.reshape(-1).astype(np.uint32), subs)


def vertex_normals(pos, faces):
    fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n


def box_uv(pos, nrm):
    """Box projection by the dominant normal axis (meshes without UVs)."""
    axis = np.argmax(np.abs(nrm), 1)
    uv = np.where(axis[:, None] == 0, pos[:, [2, 1]], np.where(axis[:, None] == 1, pos[:, [0, 2]], pos[:, [0, 1]]))
    return uv * 0.5


def vertex_tangents(pos, nrm, uv, faces):
    p0, p1, p2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    u0, u1, u2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, u1 - u0, u2 - u0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-12, 1.0 / np.where(np.abs(det) > 1e-12, det, 1.0), 0.0)
    dpdu = (e1 * d2[:, 1:] - e2 * d1[:, 1:]) * r[:, None]
    t = np.zeros_like(pos)
    for k in range(3):
        np.add.at(t, faces[:, k], dpdu)
    t -= nrm * np.sum(nrm * t, 1, keepdims=True)
    bad = np.linalg.norm(t, axis=1) < 1e-9
    if bad.any():  # no UV derivative: any vector perpendicular to the normal
        alt = np.where(np.abs(nrm[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        t[bad] = np.cross(nrm[bad], alt[bad])
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def split_by_tile(o: MeshObject, tiles: list) -> MeshObject:
    """A mesh whose submeshes are split by UDIM tile (`tiles`: (U, V) offsets, by triangle UV centroid): material m
    of tile t becomes m * len(tiles) + t. Triangles outside the listed tiles stay on the first."""
    import dataclasses
    tri = o.idx.reshape(-1, 3)
    c = o.uv[tri].mean(1)
    key = np.floor(c[:, 0]).astype(np.int64) + 1000 * np.floor(1.0 - c[:, 1]).astype(np.int64)
    tile_of = np.zeros(len(tri), np.int64)
    for i, (u, v) in enumerate(tiles):
        tile_of[key == u + 1000 * v] = i
    parts, subs, start = [], [], 0
    for s in o.submeshes:
        rows = np.arange(s.start // 3, (s.start + s.count) // 3)
        for t in range(len(tiles)):
            sel = rows[tile_of[rows] == t]
            if len(sel):
                parts.append(tri[sel])
                subs.append(Submesh(start, len(sel) * 3, s.material * len(tiles) + t))
                start += len(sel) * 3
    idx = np.concatenate(parts).reshape(-1).astype(np.uint32) if parts else o.idx[:0]
    return dataclasses.replace(o, idx=idx, submeshes=subs)


def fit(objects) -> tuple[np.ndarray, float]:
    """Center on the origin and scale to the primitives' size; returns the transform (center, scale) so
    other meshes of the same scene (a bake's high poly and cage) can follow with `apply_fit`."""
    lo = np.min([o.pos.min(0) for o in objects], 0)
    hi = np.max([o.pos.max(0) for o in objects], 0)
    center, size = (lo + hi) / 2, float(np.max(hi - lo)) or 1.0
    k = 2.0 / size
    apply_fit(objects, (center, k))
    return center, k


def apply_fit(objects, transform):
    center, k = transform
    for o in objects:
        o.pos = ((o.pos - center) * k).astype(np.float32)


def load(path, subdivide: int = 0) -> tuple[list[MeshObject], list[str], list[dict]]:
    """Meshes of a 3D scene file in its own units (`fit` scales them to the 3D View), the material
    names their submeshes index, and the scene's cameras ({name, eye, target, fov}). `subdivide`: Catmull-Clark
    levels on the file's own polygons (FBX only: the other loaders hand back triangles, where it is not
    Catmull-Clark any more)."""
    path = Path(path)
    if subdivide and path.suffix.lower() != ".fbx":
        raise ValueError(f"{path.name}: Catmull-Clark subdivision needs the file's polygons (FBX)")
    if path.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"):
        return _load_usd(path)
    if path.suffix.lower() == ".fbx":
        return _load_fbx(path, subdivide)
    import trimesh
    # OBJ: one mesh per `o` object (trimesh merges them by default), so meshes keep their names.
    extra = {"split_objects": True} if path.suffix.lower() == ".obj" else {}
    scene = trimesh.load(str(path), force="scene", process=False, **extra)
    materials, mat_index, objects, colors = [], {}, [], []

    def material_of(geom):
        m = getattr(getattr(geom, "visual", None), "material", None)
        name = getattr(m, "name", None) or "Default material"
        if name not in mat_index:
            mat_index[name] = len(materials)
            materials.append(name)
            main = getattr(m, "main_color", None)  # trimesh's base / diffuse color (uint8 RGBA)
            colors.append([float(c) / 255.0 for c in main] if main is not None else [1.0, 1.0, 1.0, 1.0])
        return mat_index[name]

    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not hasattr(geom, "faces") or len(geom.faces) == 0:
            continue
        pos = trimesh.transform_points(geom.vertices, transform)
        # Normals only when the file has them (trimesh's own computation wants scipy): else ours.
        nrm = geom._cache["vertex_normals"] if "vertex_normals" in geom._cache else None
        if nrm is not None:
            nrm = nrm @ np.linalg.inv(transform[:3, :3])  # inverse transpose, row vectors
        uv = getattr(getattr(geom, "visual", None), "uv", None)
        if uv is not None:
            uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])  # OBJ/glTF V up -> image rows down
        m = material_of(geom)
        o = finish(str(node), pos, nrm, uv, geom.faces, np.full(len(geom.faces), m), max(len(materials), 1))
        visual = getattr(geom, "visual", None)
        if getattr(visual, "kind", None) == "vertex":
            o.colors = np.asarray(visual.vertex_colors, np.float32) / 255.0
        objects.append(o)
    if not objects:
        raise ValueError(f"{path.name}: no triangle meshes")
    for o in objects:  # materials counted after all objects
        o.material_colors = colors or [[1.0, 1.0, 1.0, 1.0]]
        for s in o.submeshes:
            s.material = min(s.material, len(materials) - 1)
    cams = []
    if getattr(scene, "has_camera", False):
        try:
            t = scene.camera_transform
            eye = t[:3, 3]
            cams.append({"name": scene.camera.name or "Camera", "eye": list(map(float, eye)),
                         "target": list(map(float, eye - t[:3, 2])), "fov": float(scene.camera.fov[1])})
        except Exception:  # noqa: BLE001 - a malformed camera is skipped
            pass
    return objects, materials or ["Default material"], cams


def _load_fbx(path, subdivide=0):
    """FBX through ufbx (trafilatura3d/fbx.py): meters, Y up; faces without a material get "Default material"."""
    from trafilatura3d import fbx
    meshes, materials, cams, colors = fbx.read(path, subdivide)
    used_default = any((m["material"] >= len(materials)).any() for m in meshes)
    if used_default or not materials:
        materials = materials + ["Default material"]
        colors = colors + [[1.0, 1.0, 1.0, 1.0]]
    objects = []
    for m in meshes:
        groups = np.minimum(m["material"], len(materials) - 1)
        uv = None if m["uv"] is None else np.column_stack([m["uv"][:, 0], 1.0 - m["uv"][:, 1]])  # V up -> rows down
        o = finish(m["name"], m["pos"], m["nrm"], uv, m["faces"], groups, len(materials))
        o.visible, o.colors, o.material_colors = m["visible"], m["colors"], colors
        objects.append(o)
    if not objects:
        raise ValueError(f"{Path(path).name}: no meshes")
    return objects, materials, cams


def _load_usd(path):
    from pxr import Usd, UsdGeom, UsdShade
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    materials, mat_index, objects, cams, colors = [], {}, [], [], []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = np.array(mesh.GetPointsAttr().Get() or [], np.float64)
            counts = np.array(mesh.GetFaceVertexCountsAttr().Get() or [], np.int64)
            ids = np.array(mesh.GetFaceVertexIndicesAttr().Get() or [], np.int64)
            if len(pts) == 0 or len(counts) == 0:
                continue
            m = np.array(cache.GetLocalToWorldTransform(prim), np.float64)
            pts = pts @ m[:3, :3] + m[3, :3]
            faces, starts = [], np.concatenate([[0], np.cumsum(counts)[:-1]])
            for s, c in zip(starts, counts):  # fan triangulation
                for k in range(1, c - 1):
                    faces.append([ids[s], ids[s + k], ids[s + k + 1]])
            uv = None
            primvars = UsdGeom.PrimvarsAPI(prim)
            for name in ("st", "uv", "UVMap"):
                pv = primvars.GetPrimvar(name)
                if pv and pv.HasValue() and pv.GetInterpolation() in ("vertex", "varying"):
                    uv = np.array(pv.Get(), np.float64)
                    uv = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])
                    break
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            name = mat.GetPrim().GetName() if mat else "Default material"
            if name not in mat_index:
                mat_index[name] = len(materials)
                materials.append(name)
                colors.append(_usd_material_color(mat))
            o = finish(prim.GetName(), pts, None, uv, faces, np.full(len(faces), mat_index[name]), len(materials))
            dc = mesh.GetDisplayColorPrimvar()
            if dc and dc.HasValue() and dc.GetInterpolation() in ("vertex", "varying"):
                rgb = np.array(dc.Get(), np.float32)
                if len(rgb) == len(pts):
                    o.colors = np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], 1)
            objects.append(o)
        elif prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim).GetCamera()
            m = np.array(cache.GetLocalToWorldTransform(prim), np.float64)
            eye = m[3, :3]
            fov = math.degrees(2 * math.atan(cam.verticalAperture / (2 * cam.focalLength)))
            cams.append({"name": prim.GetName(), "eye": list(map(float, eye)),
                         "target": list(map(float, eye - m[2, :3])), "fov": fov})
    if not objects:
        raise ValueError(f"{path.name}: no meshes")
    for o in objects:
        o.material_colors = colors or [[1.0, 1.0, 1.0, 1.0]]
    return objects, materials or ["Default material"], cams


def _usd_material_color(mat) -> list:
    """A UsdPreviewSurface's diffuseColor (white without one)."""
    if mat:
        from pxr import UsdShade
        for prim in mat.GetPrim().GetChildren():
            shader = UsdShade.Shader(prim)
            if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
                value = shader.GetInput("diffuseColor").Get() if shader.GetInput("diffuseColor") else None
                if value is not None:
                    return [float(value[0]), float(value[1]), float(value[2]), 1.0]
    return [1.0, 1.0, 1.0, 1.0]

