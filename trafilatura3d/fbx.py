"""FBX import through ufbx (https://github.com/ufbx/ufbx, MIT), built with native/ufbx_shim.c by
tools/build_ufbx.py into native/<platform>/ and called through ctypes. Scenes come in meters, right
handed, Y up, every mesh instance triangulated in world space; `subdivide` Catmull-Clark levels first, on the
file's own polygons (a smooth high poly for the bakers)."""

from __future__ import annotations

import ctypes
import platform
import sys
from pathlib import Path

import numpy as np

NATIVE = Path(__file__).resolve().parent / "native"
_lib = None


def library_path() -> Path:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        folder, name = "macos-" + ("arm64" if machine == "arm64" else "x86_64"), "libtrafilatura3d_ufbx.dylib"
    elif sys.platform == "win32":
        folder, name = "windows-x64", "trafilatura3d_ufbx.dll"
    else:
        folder, name = "linux-" + machine, "libtrafilatura3d_ufbx.so"
    return NATIVE / folder / name


def available() -> bool:
    return library_path().exists()


def _library():
    global _lib
    if _lib is None:
        if not available():
            raise OSError(f"FBX support is not built for this platform: run tools/build_ufbx.py ({library_path()})")
        lib = ctypes.CDLL(str(library_path()))
        P, S, U32, F = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)
        sig = {
            "tf_load": (P, [ctypes.c_char_p, U32, ctypes.c_char_p, S]), "tf_free": (None, [P]),
            "tf_mesh_count": (S, [P]), "tf_mesh_name": (ctypes.c_char_p, [P, S]), "tf_mesh_visible": (ctypes.c_int, [P, S]),
            "tf_mesh_has_uv": (ctypes.c_int, [P, S]), "tf_mesh_corners": (U32, [P, S]),
            "tf_mesh_data": (None, [P, S, F, F, F, ctypes.POINTER(U32)]),
            "tf_mesh_has_color": (ctypes.c_int, [P, S]), "tf_mesh_colors": (None, [P, S, F]),
            "tf_material_count": (S, [P]), "tf_material_name": (ctypes.c_char_p, [P, S]),
            "tf_material_color": (None, [P, S, F]),
            "tf_camera_count": (S, [P]), "tf_camera": (ctypes.c_char_p, [P, S, F]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        _lib = lib
    return _lib


def _ptr(a, kind=ctypes.c_float):
    return a.ctypes.data_as(ctypes.POINTER(kind))


def read(path, subdivide: int = 0) -> tuple[list[dict], list[str], list[dict], list]:
    """Meshes ({name, visible, pos, nrm, uv or None, colors or None, faces, material per face}), material
    names, cameras ({name, eye, target, fov}) and material base colors (RGBA). Corners are welded back
    into indexed vertices. `subdivide`: Catmull-Clark levels applied to every mesh first."""
    lib = _library()
    err = ctypes.create_string_buffer(1024)
    scene = lib.tf_load(str(path).encode(), int(subdivide), err, len(err))
    if not scene:
        raise ValueError(f"{Path(path).name}: {err.value.decode(errors='replace').strip()}")
    try:
        materials = [(lib.tf_material_name(scene, i) or b"").decode(errors="replace") or f"material{i}"
                     for i in range(lib.tf_material_count(scene))]
        colors = []
        rgba = (ctypes.c_float * 4)()
        for i in range(len(materials)):
            lib.tf_material_color(scene, i, rgba)
            colors.append([float(x) for x in rgba])
        meshes = []
        for i in range(lib.tf_mesh_count(scene)):
            n = lib.tf_mesh_corners(scene, i)
            pos, nrm = np.empty((n, 3), np.float32), np.empty((n, 3), np.float32)
            uv, mat = np.empty((n, 2), np.float32), np.empty(n // 3, np.uint32)
            col = np.empty((n, 4), np.float32)
            lib.tf_mesh_data(scene, i, _ptr(pos), _ptr(nrm), _ptr(uv), _ptr(mat, ctypes.c_uint32))
            lib.tf_mesh_colors(scene, i, _ptr(col))
            has_uv, has_color = bool(lib.tf_mesh_has_uv(scene, i)), bool(lib.tf_mesh_has_color(scene, i))
            # Weld identical corners (position, normal, UV, color): smooth tangents and shared vertices.
            key = np.concatenate([pos, nrm, uv, col], 1)
            unique, inverse = np.unique(key, axis=0, return_inverse=True)
            meshes.append({"name": (lib.tf_mesh_name(scene, i) or b"").decode(errors="replace") or f"mesh{i}",
                           "visible": bool(lib.tf_mesh_visible(scene, i)),
                           "pos": unique[:, :3], "nrm": unique[:, 3:6], "uv": unique[:, 6:8] if has_uv else None,
                           "colors": unique[:, 8:12] if has_color else None,
                           "faces": inverse.reshape(-1, 3), "material": mat.astype(np.int64)})
        cams = []
        out = (ctypes.c_float * 7)()
        for i in range(lib.tf_camera_count(scene)):
            name = (lib.tf_camera(scene, i, out) or b"Camera").decode(errors="replace")
            if name.startswith("Producer "):  # FBX 6's built-in viewport cameras, not the scene's
                continue
            eye, forward = np.array(out[0:3]), np.array(out[3:6])
            cams.append({"name": name, "eye": eye.tolist(), "target": (eye + forward).tolist(), "fov": float(out[6])})
        return meshes, materials, cams, colors
    finally:
        lib.tf_free(scene)
