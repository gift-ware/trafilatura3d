"""The .spp container: an HDF5 file of byte datasets (the document, the mesh, the resources, the cached channels).

Images (`resources/<hash>.image`, `texture-cache/...`) are a header, a Qt `QVariantMap` written by `QDataStream`
(big endian: uint32 count, then per entry the key as UTF-16 with its byte length and the value as a QVariant: uint32
type, one null byte, the value), then the pixels, a raw LZ4 block with "compression" = "lz4" (of "dataSizeLZ4" bytes, or after its big-endian uint32
size when the header has no such key). A UDIM image is a list of its tiles' images (`tile_list`).
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

from trafilatura3d.spp.document import parse

# OpenGL internal formats of the cached channel images: (numpy type, components)
GL_FORMATS = {
    0x8C43: (np.uint8, 4),  # SRGB8_ALPHA8
    0x8058: (np.uint8, 4),  # RGBA8
    0x822B: (np.uint8, 2),  # RG8
    0x8229: (np.uint8, 1),  # R8
    0x822F: (np.float16, 2),  # RG16F
    0x881A: (np.float16, 4),  # RGBA16F
    0x822D: (np.float16, 1),  # R16F
    0x805B: (np.uint16, 4),  # RGBA16
    0x822C: (np.uint16, 2),  # RG16
    0x822A: (np.uint16, 1),  # R16
    0x8814: (np.float32, 4),  # RGBA32F
    0x8230: (np.float32, 2),  # RG32F
    0x822E: (np.float32, 1),  # R32F
}
# Named formats of the resource images: (numpy type, components, blue first)
NAMED_FORMATS = {"LUM8": (np.uint8, 1, False), "LUM16": (np.uint16, 1, False), "LUM32F": (np.float32, 1, False),
                 "BGRA8": (np.uint8, 4, True), "RGBA8": (np.uint8, 4, False), "RGBA16": (np.uint16, 4, False),
                 "RGBA16F": (np.float16, 4, False), "RGBA32F": (np.float32, 4, False)}
# Older records name the format by a number; the pixel size tells it
BY_PIXEL_SIZE = {1: (np.uint8, 1, False), 2: (np.uint16, 1, False), 4: (np.uint8, 4, True),
                 8: (np.uint16, 4, False), 16: (np.float32, 4, False)}


class ContainerError(ValueError):
    pass


def _qmap(b: bytes, o: int = 0) -> tuple[dict, int]:
    n, = struct.unpack_from(">I", b, o)
    o += 4
    out = {}
    for _ in range(n):
        size, = struct.unpack_from(">I", b, o)
        o += 4
        key = b[o:o + size].decode("utf-16-be")
        o += size
        kind, = struct.unpack_from(">I", b, o)
        o += 5  # the type and the null flag
        if kind in (2, 3):  # int, uint
            value, = struct.unpack_from(">i" if kind == 2 else ">I", b, o)
            o += 4
        elif kind in (4, 5):  # long long, unsigned long long
            value, = struct.unpack_from(">q" if kind == 4 else ">Q", b, o)
            o += 8
        elif kind == 1:  # bool
            value = bool(b[o])
            o += 1
        elif kind == 10:  # string
            size, = struct.unpack_from(">I", b, o)
            o += 4
            if size == 0xFFFFFFFF:
                value = ""
            else:
                value = b[o:o + size].decode("utf-16-be")
                o += size
        else:
            raise ContainerError(f"image header: unknown value type {kind} for {key!r}")
        out[key] = value
    return out, o


def image_header(data: bytes) -> dict:
    return _qmap(data)[0]


def decode_image(data: bytes) -> tuple[np.ndarray, dict]:
    """(pixels (H, W, C), rows from the top, RGBA order; the header) of an image record. 8 and 16-bit integers become
    float32 0-1, half floats stay half."""
    import lz4.block
    head, o = _qmap(data)
    w, h = int(head["width"]), int(head["height"])
    bgr = False
    if "internalFormat" in head:
        fmt = int(head["internalFormat"])
        if fmt not in GL_FORMATS:
            raise ContainerError(f"unknown image format {fmt:#x}")
        dtype, comps = GL_FORMATS[fmt]
    elif isinstance(head.get("format"), str):
        if head["format"] not in NAMED_FORMATS:
            raise ContainerError(f"unknown image format {head['format']!r}")
        dtype, comps, bgr = NAMED_FORMATS[head["format"]]
    else:
        px = int(head["pitch"]) // w
        if px not in BY_PIXEL_SIZE:
            raise ContainerError(f"unknown image format {head.get('format')!r} ({px} bytes a pixel)")
        dtype, comps, bgr = BY_PIXEL_SIZE[px]
    row = w * comps * np.dtype(dtype).itemsize
    pitch = int(head.get("pitch") or row)
    size = int(head.get("bufferSize") or head.get("dataSizeRaw") or h * pitch)
    if head.get("compression") == "lz4":
        if "dataSizeLZ4" in head:  # the block follows the header
            n = int(head["dataSizeLZ4"])
        else:  # the block's size first
            n, = struct.unpack_from(">I", data, o)
            o += 4
        raw = lz4.block.decompress(data[o:o + n], uncompressed_size=size)
    else:
        raw = data[o:o + size]
    rows = np.frombuffer(raw, np.uint8, h * pitch).reshape(h, pitch)[:, :row]
    a = np.ascontiguousarray(rows).view(dtype).reshape(h, w, comps)
    if bgr:
        a = a[..., [2, 1, 0, 3]]
    if dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    return a, head


class Container:
    """An open .spp file."""

    def __init__(self, path):
        import h5py
        self.path = Path(path)
        try:
            self.file = h5py.File(self.path, "r")
        except OSError as e:
            raise ContainerError(f"{self.path.name}: not a .spp project ({e})") from None
        if "paint/document.bin" not in self.file:
            raise ContainerError(f"{self.path.name}: no project document")

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def bytes(self, name: str) -> bytes:
        return bytes(self.file[name][()])

    def has(self, name: str) -> bool:
        return name in self.file

    def document(self):
        return parse(self.bytes("paint/document.bin"))

    def resource(self, url: str) -> str | None:
        """The dataset of a resource url (`.../<name>?version=<hash>.<kind>`), or None when the file lacks it."""
        m = re.search(r"version=([0-9a-f]{40})\.(\w+)", url)
        if m is None:
            return None
        name = f"resources/{m.group(1)}.{m.group(2)}"
        return name if name in self.file else None

    def image(self, dataset: str) -> dict:
        """The tiles of an image resource: {UDIM, or 0 for a single image: (pixels, header)}."""
        data = self.bytes(dataset)
        if not data.startswith(TILE_LIST):
            return {0: decode_image(data)}
        return {udim: decode_image(self.bytes(f"resources/{name}")) for udim, name in tile_list(data).items()}


TILE_LIST = b".S4TID.."  # a UDIM image: its tiles' images


def tile_list(data: bytes) -> dict:
    """UDIM -> image dataset name of a tile list: the magic, uint32 version, uint32 (0), uint32 count, then per tile
    uint32 u, uint32 v, the name (uint32 length + bytes); big endian."""
    _version, _zero, n = struct.unpack_from(">3I", data, len(TILE_LIST))
    o = len(TILE_LIST) + 12
    out = {}
    for _ in range(n):
        u, v, size = struct.unpack_from(">3I", data, o)
        o += 12
        out[1001 + u + 10 * v] = data[o:o + size].decode("ascii")
        o += size
    return out
