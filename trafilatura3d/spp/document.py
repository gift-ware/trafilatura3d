"""The project document of a .spp file (`paint/document.bin`, `paint/default_material.bin`): a tagged binary tree.

Layout, little endian: a 12-byte header, then one value. A value is a tag byte and its payload:

    0x01 float32 · 0x02 2 x float32 · 0x03 3 x float32 · 0x05 int32 · 0x06 2 x int32 · 0x09 int64 · 0x0a bool (1 byte)
    0x0b int32 · 0x0c uint64 · 0x0d 9 x float32 · 0x0e 16 x float32 · 0x0f uint64 · 0x10 string (uint32 length, UTF-8)
    0x15 uint64 (channel bits)
    0x11, 0x13 list: uint32 end offset, uint32 count, the values
    0x12, 0x14 object: one byte (0xff = none), uint32 end offset, the class name (string), uint16 field count, then
               per field its name (string) and its value

End offsets are absolute in the blob: the reader checks every object and list ends where it says.
"""

from __future__ import annotations

import struct

HEADER = 12

_FIXED = {0x01: "<f", 0x02: "<2f", 0x03: "<3f", 0x05: "<i", 0x06: "<2i", 0x09: "<q", 0x0b: "<i", 0x0c: "<Q",
          0x0d: "<9f", 0x0e: "<16f", 0x0f: "<Q", 0x15: "<Q"}
_FIXED = {t: (struct.Struct(f), struct.calcsize(f)) for t, f in _FIXED.items()}
CLASS = "@"  # the key holding an object's class name


class DocumentError(ValueError):
    pass


class _Reader:
    def __init__(self, data: bytes):
        self.b = data
        self.o = 0

    def u32(self) -> int:
        v, = struct.unpack_from("<I", self.b, self.o)
        self.o += 4
        return v

    def string(self) -> str:
        n = self.u32()
        v = self.b[self.o:self.o + n]
        self.o += n
        return v.decode("utf-8", "replace")

    def end(self, end: int, what: str) -> None:
        if self.o != end:
            raise DocumentError(f"{what} ends at {self.o:#x}, its header says {end:#x}")

    def value(self):
        t = self.b[self.o]
        self.o += 1
        if t in (0x12, 0x14):
            if self.b[self.o] == 0xFF:
                self.o += 1
                return None
            self.o += 1
            end = self.u32()
            out = {CLASS: self.string()}
            n, = struct.unpack_from("<H", self.b, self.o)
            self.o += 2
            for _ in range(n):
                key = self.string()
                out[key] = self.value()
            self.end(end, out[CLASS])
            return out
        if t in (0x11, 0x13):
            end = self.u32()
            n = self.u32()
            out = [self.value() for _ in range(n)]
            self.end(end, "a list")
            return out
        if t == 0x10:
            return self.string()
        if t == 0x0A:
            v = bool(self.b[self.o])
            self.o += 1
            return v
        fixed = _FIXED.get(t)
        if fixed is None:
            raise DocumentError(f"unknown tag {t:#x} at {self.o - 1:#x}")
        s, n = fixed
        v = s.unpack_from(self.b, self.o)
        self.o += n
        return v[0] if len(v) == 1 else list(v)


def parse(data: bytes):
    """The tree of a document blob: objects as dicts (class name under CLASS), lists, numbers, strings."""
    r = _Reader(data)
    r.o = HEADER
    try:
        root = r.value()
    except (struct.error, IndexError) as e:
        raise DocumentError(f"truncated document ({e})") from None
    if r.o != len(data):
        raise DocumentError(f"{len(data) - r.o} bytes after the document")
    return root


def walk(node, cls: str | None = None):
    """Every object below `node` (itself included), depth first; only those of class `cls` if given."""
    stack = [node]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if cls is None or v.get(CLASS) == cls:
                yield v
            stack.extend(reversed([x for x in v.values() if isinstance(x, (dict, list))]))
        elif isinstance(v, list):
            stack.extend(reversed([x for x in v if isinstance(x, (dict, list))]))
