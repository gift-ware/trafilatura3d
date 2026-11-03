"""The project document of a .spp file (`paint/document.bin`, `paint/default_material.bin`): a binary tree, in two
versions (the uint32 after the header's first one; the autosaves of newer versions write 1).

Version 0, little endian: a 12-byte header, then one value. A value is a tag byte and its payload:

    0x01 float32 · 0x02 2 x float32 · 0x03 3 x float32 · 0x05 int32 · 0x06 2 x int32 · 0x09 int64 · 0x0a bool (1 byte)
    0x0b int32 · 0x0c uint64 · 0x0d 9 x float32 · 0x0e 16 x float32 · 0x0f uint64 · 0x10 string (uint32 length, UTF-8)
    0x15 uint64 (channel bits)
    0x11, 0x13 list: uint32 end offset, uint32 count, the values
    0x12, 0x14 object: one byte (0xff = none), uint32 end offset, the class name (string), uint16 field count, then
               per field its name (string) and its value

End offsets are absolute in the blob: the reader checks every object and list ends where it says.

Version 1: the 12-byte header, the root's type (uint32, an object) and the count of classes, then the root. Values
carry no tag: an object's fields declare their types once, with its class (the type numbers are version 0's tags;
9 and 11 are int32 here, 0x16 a 128-bit channel mask, bools are 0 or 0xff):

    object (0x12, 0x14): uint32 end offset (0 = none), its class, the values of the class's fields in order
    class: int32 index of a class met before, or -1 then its name (string), uint32 field count, and per field its
           name (string) and type (uint32)
    0x13 list: uint32 count, the objects
    0x11 list: uint32 count, then (count > 0) one class and the values of each object of that class, no end offsets

The reader gives the version 0 tree: channel masks as their low 64 bits, `sourceTransparent` (one source here) as
a list, the channel of a channel or of a procedural input (a bit index here, -1 none) as its bit (`BIT_INDEX`).
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


_FIXED_V1 = {0x01: "<f", 0x02: "<2f", 0x03: "<3f", 0x04: "<4f", 0x05: "<i", 0x06: "<2i", 0x07: "<3i", 0x08: "<4i",
             0x09: "<i", 0x0b: "<i", 0x0c: "<Q", 0x0d: "<9f", 0x0e: "<16f", 0x0f: "<Q", 0x15: "<Q", 0x16: "<2Q"}
_FIXED_V1 = {t: (struct.Struct(f), struct.calcsize(f)) for t, f in _FIXED_V1.items()}
CHANNEL_MASK = 0x16
BIT_INDEX = {("DataChannel", "type"), ("DataProceduralInput", "channelType")}  # version 1 index -> version 0 bit


class _ReaderV1(_Reader):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.classes: list[tuple[str, list[tuple[str, int]]]] = []

    def i32(self) -> int:
        v, = struct.unpack_from("<i", self.b, self.o)
        self.o += 4
        return v

    def klass(self):
        ref = self.i32()
        if ref != -1:
            if not 0 <= ref < len(self.classes):
                raise DocumentError(f"class {ref} at {self.o - 4:#x}: not declared before")
            return self.classes[ref]
        name = self.string()
        fields = [(self.string(), self.u32()) for _ in range(self.u32())]
        self.classes.append((name, fields))
        return self.classes[-1]

    def fields(self, cls) -> dict:
        name, fields = cls
        out = {CLASS: name}
        for key, t in fields:
            out[key] = self.typed(t)
        if isinstance(out.get("sourceTransparent"), dict):  # version 0 keeps a list of sources there
            out["sourceTransparent"] = [out["sourceTransparent"]]
        for cls, key in BIT_INDEX:
            if name == cls and isinstance(out.get(key), int):
                out[key] = 0 if out[key] < 0 else 1 << out[key]
        return out

    def typed(self, t: int):
        if t in (0x12, 0x14):
            end = self.u32()
            if end == 0:
                return None
            out = self.fields(self.klass())
            self.end(end, out[CLASS])
            return out
        if t == 0x13:
            return [self.typed(0x12) for _ in range(self.u32())]
        if t == 0x11:
            n = self.u32()
            if n == 0:
                return []
            cls = self.klass()
            return [self.fields(cls) for _ in range(n)]
        if t == 0x10:
            return self.string()
        if t == 0x0A:
            v = self.b[self.o] != 0
            self.o += 1
            return v
        fixed = _FIXED_V1.get(t)
        if fixed is None:
            raise DocumentError(f"unknown type {t:#x} at {self.o:#x}")
        s, n = fixed
        v = s.unpack_from(self.b, self.o)
        self.o += n
        if t == CHANNEL_MASK:
            return v[0]  # the low 64 bits: the channels version 0 knows
        return v[0] if len(v) == 1 else list(v)

    def root(self):
        t = self.u32()
        self.u32()  # the count of classes the document declares
        return self.typed(t)


def version(data: bytes) -> int:
    v, = struct.unpack_from("<I", data, 4)
    return v


def parse(data: bytes):
    """The tree of a document blob: objects as dicts (class name under CLASS), lists, numbers, strings."""
    if len(data) < HEADER:
        raise DocumentError("truncated document (no header)")
    ver = version(data)
    if ver not in (0, 1):
        raise DocumentError(f"document version {ver}: not supported (versions 0 and 1)")
    r = _Reader(data) if ver == 0 else _ReaderV1(data)
    r.o = HEADER
    try:
        root = r.value() if ver == 0 else r.root()
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
