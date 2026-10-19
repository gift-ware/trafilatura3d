"""The mesh of a .spp project (`paint/scene.assbin`): a binary scene dump in the layout of the open Assimp format
("ASSIMP.binary-dump."), version 1.6.

A 512-byte header, then chunks; a chunk is a uint64 size (counting its own 8 bytes) and its body:

    scene     uint32 flags, the root node chunk, uint32 mesh count, the mesh chunks, uint32 material count, the
              material chunks, then four uint32 counts of what is not read (animations, textures, lights, cameras: 0)
    node      name, 4x4 matrix, ..., child node chunks, then the metadata: uint16 1, uint32 n, n keys, uint32 n, n
              typed values (uint32 type: 1 int32, 3 float, 5 3 x float, 6 4x4 float). The root's metadata gives the normalization
              the vertices went through: position = (file position - ALG_PRE_NORMALIZATION_CENTER) / RADIUS
    mesh      name, uint32 primitive types, uint32 material index, uint32 components (bits: 1 positions, 2 normals,
              4 tangents and bitangents, 0x100 << n UV set n), uint32 vertex count, the positions (3 x float), each
              other array after its uint32 count (a UV set after its uint32 component count too), uint32 triangle
              count, uint32 (1), per triangle uint32 3, the three indices, uint32 (a flag)
    material  properties; "$mat.uid" holds the material's uid (the project's Texture Set refers to it)

Every chunk is checked to end where its size says.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"ASSIMP.binary-dump."
HEADER = 512
_KEY = re.compile(rb"ALG_PRE_NORMALIZATION_CENTER")


class SceneError(ValueError):
    pass


@dataclass
class Mesh:
    name: str
    material: int
    pos: np.ndarray  # (n, 3), normalized
    nrm: np.ndarray
    uv: np.ndarray  # (n, 2), V up
    tri: np.ndarray  # (m, 3)


@dataclass
class Scene:
    meshes: list
    materials: list  # uid per material index
    center: np.ndarray  # the normalization: file position = position * radius + center
    radius: float


class _Reader:
    def __init__(self, data: bytes, o: int):
        self.b, self.o = data, o

    def u32(self) -> int:
        v, = struct.unpack_from("<I", self.b, self.o)
        self.o += 4
        return v

    def u64(self) -> int:
        v, = struct.unpack_from("<Q", self.b, self.o)
        self.o += 8
        return v

    def string(self) -> str:
        n = self.u32()
        v = self.b[self.o:self.o + n]
        self.o += n
        return v.decode("utf-8", "replace")

    def floats(self, n: int) -> np.ndarray:
        v = np.frombuffer(self.b, "<f4", n, self.o)
        self.o += 4 * n
        return v


def _metadata(data: bytes, o: int) -> dict:
    r = _Reader(data, o)
    n = r.u32()
    keys = [r.string() for _ in range(n)]
    if r.u32() != n:
        raise SceneError("metadata: the values are not as many as the keys")
    out = {}
    for key in keys:
        kind = r.u32()
        if kind == 1:
            out[key] = struct.unpack_from("<i", data, r.o)[0]
            r.o += 4
        elif kind == 3:
            out[key] = float(r.floats(1)[0])
        elif kind == 5:
            out[key] = r.floats(3).astype(np.float64)
        elif kind == 6:
            out[key] = r.floats(16).astype(np.float64).reshape(4, 4)
        else:
            raise SceneError(f"metadata {key!r}: unknown type {kind}")
    return out


def _mesh(r: _Reader, end: int) -> Mesh:
    name = r.string()
    prim, material, comps, nv = r.u32(), r.u32(), r.u32(), r.u32()
    if prim != 4:
        raise SceneError(f"{name}: not a triangle mesh ({prim})")
    if not comps & 1:
        raise SceneError(f"{name}: no positions")
    pos = r.floats(3 * nv).reshape(nv, 3)
    nrm = None
    for bit in (2, 4, 4):  # normals, tangents, bitangents
        if comps & bit:
            if r.u32() != nv:
                raise SceneError(f"{name}: vertex array of another length")
            a = r.floats(3 * nv).reshape(nv, 3)
            if bit == 2 and nrm is None:
                nrm = a
    uvs = []
    for n in range(8):
        if comps & (0x100 << n):
            _components, count = r.u32(), r.u32()
            if count != nv:
                raise SceneError(f"{name}: UV set of another length")
            uvs.append(r.floats(3 * nv).reshape(nv, 3)[:, :2])
    if comps & 0xFFFF0000:
        raise SceneError(f"{name}: vertex colors are not read")
    nf, _one = r.u32(), r.u32()
    f = np.frombuffer(r.b, "<u4", 5 * nf, r.o).reshape(nf, 5)
    r.o += 20 * nf
    if (f[:, 0] != 3).any():
        raise SceneError(f"{name}: a face is not a triangle")
    if r.o != end:
        raise SceneError(f"{name}: the mesh ends at {r.o:#x}, its size says {end:#x}")
    if not uvs:
        raise SceneError(f"{name}: no UVs")
    return Mesh(name, material, pos.astype(np.float64), None if nrm is None else nrm.astype(np.float64),
                uvs[0].astype(np.float64), f[:, 1:4].astype(np.int64))


def parse(data: bytes) -> Scene:
    if not data.startswith(MAGIC):
        raise SceneError("not a binary scene dump")
    major, minor = struct.unpack_from("<2I", data, 44)
    shortened, compressed = struct.unpack_from("<2H", data, 60)
    if major != 1 or shortened or compressed:
        raise SceneError(f"unsupported scene dump {major}.{minor} (shortened {shortened}, compressed {compressed})")
    r = _Reader(data, HEADER)
    if HEADER + r.u64() != len(data):
        raise SceneError("the scene's size is not the file's")
    r.u32()  # flags
    root = r.o
    root_end = root + r.u64()
    m = _KEY.search(data, root, root_end)
    if m is None:
        raise SceneError("the root node has no normalization")
    # the metadata block: uint16 1, uint32 count, the keys; the first key is one of the root's six
    start = data.rfind(b"\x01\x00", root, m.start())
    meta = None
    while start > root:
        try:
            meta = _metadata(data, start + 2)
            break
        except (SceneError, struct.error, UnicodeDecodeError):
            start = data.rfind(b"\x01\x00", root, start)
    if meta is None or "ALG_PRE_NORMALIZATION_RADIUS" not in meta:
        raise SceneError("the root node's metadata cannot be read")
    r.o = root_end
    meshes = []
    for _ in range(r.u32()):
        at = r.o
        size = r.u64()
        meshes.append(_mesh(r, at + size))
    count = r.u32()
    materials = []
    for _ in range(count):
        at = r.o
        size = r.u64()
        body = data[r.o:at + size]
        k = body.find(b"$mat.uid")
        if k < 0:
            raise SceneError("a material without uid")
        # key, uint32 semantic, uint32 index, uint32 type, uint32 length, then the value
        materials.append(struct.unpack_from("<Q", body, k + 8 + 16)[0])
        r.o = at + size
    others = [r.u32() for _ in range(4)]  # animations, textures, lights, cameras
    if any(others) or r.o != len(data):
        raise SceneError("the scene has animations, textures, lights or cameras, or more after them")
    return Scene(meshes, materials, np.asarray(meta["ALG_PRE_NORMALIZATION_CENTER"], np.float64),
                 float(meta["ALG_PRE_NORMALIZATION_RADIUS"]))
