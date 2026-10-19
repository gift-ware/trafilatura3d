"""Painted pixels on the GPU: per (layer uid, channel, UDIM tile) of a Paint layer and per (layer uid, "mask", tile)
of a painted mask, RGB = the painted value, A = how much is painted (the Eraser lowers it). A mask, a gray value,
is stored in two channels (value, how much is painted) where the GPU writes them; read whole, every entry is RGBA.

The pixels are stored sparsely (`Pages`): the tile is cut into square pages and only the pages a stroke reached
exist, so a layer painted in one spot holds that spot, not a whole texture per channel and tile.

Entries keep their own size: a Texture Set resized later reads them resampled, and the next stroke resamples
them to the new size. Every change bumps the entry's revision, part of the stack jobs' keys.
"""

from __future__ import annotations

import itertools
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wgpu

from trafilatura3d.paint.channels import FORMATS

MASK = "mask"
# Storage per channel depth (8-bit channels in 8 bits, the rest in float).
STORAGE = {"8": "rgba8unorm", "16": "rgba32float", "16f": "rgba16float", "32f": "rgba32float"}
# Value and how much is painted, 8 bits: what is stored between the strokes (a stroke accumulates in float and is
# rounded once when laid; the mask's stack computes in half floats)
MASK_STORAGE = "rg8unorm"
FALLBACK = {"rg16float": "rgba16float", "rg8unorm": "rgba8unorm"}  # without the adapter's formats (COMPACT_FEATURE)
_REVISIONS = itertools.count(1)  # entries' revisions (PaintStore._bump)
DTYPE = {"rgba8unorm": (np.uint8, 4), "rgba16float": (np.float16, 8), "rgba32float": (np.float32, 16),
         "rg16float": (np.float16, 4), "rg8unorm": (np.uint8, 2)}
USAGE = (wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
         | wgpu.TextureUsage.COPY_DST)
PAGE = 256  # the side of a page (larger past MAX_PAGES a side); the brush saves and lays its strokes by page
MAX_PAGES = 16  # a side: a tile has at most 256 pages, the layers a texture array has by default
EXPAND = (Path(__file__).parent / "wgsl" / "expand.wgsl").read_text(encoding="utf-8")
_EXPAND_PIPES = {}  # (device, destination format) -> expand.wgsl pipeline


def storage_for(fmt: str | None) -> str:
    """The texture format of a channel format (None: a mask)."""
    return MASK_STORAGE if fmt is None else STORAGE[FORMATS[fmt][1]]


def supported(device, fmt: str) -> str:
    """The storage format a device writes for `fmt`: two channel formats need the adapter's formats."""
    from trafilatura3d.evaluator import COMPACT_FEATURE
    return FALLBACK[fmt] if fmt in FALLBACK and COMPACT_FEATURE not in device.features else fmt


def two(fmt: str) -> bool:
    """A two channel storage: value and how much is painted."""
    return fmt in ("rg16float", "rg8unorm")


def unpacked(fmt: str) -> str:
    """The RGBA format a two channel storage is read whole into."""
    return {"rg16float": "rgba16float", "rg8unorm": "rgba8unorm"}.get(fmt, fmt)


def page_grid(size) -> tuple[int, int, int]:
    """(page side, pages per row, rows) of a tile of `size`."""
    w, h = size
    side = max(PAGE, -(-max(w, h) // MAX_PAGES))
    return side, -(-w // side), -(-h // side)


def _array(device, side: int, layers: int, fmt: str):
    return device.create_texture(size=(side, side, layers), format=fmt,
                                 usage=USAGE | wgpu.TextureUsage.RENDER_ATTACHMENT)


def expand_pipe(device, src: str, dst: str):
    """expand.wgsl from pages of format `src` into a texture of format `dst` (two channels unpacked or packed)."""
    key = (device, src, dst)
    if key not in _EXPAND_PIPES:
        code = (EXPAND.replace("FORMAT", dst).replace("SRC_TWO", str(two(src)).lower())
                .replace("DST_TWO", str(two(dst)).lower()))
        _EXPAND_PIPES[key] = device.create_compute_pipeline(layout="auto", compute={
            "module": device.create_shader_module(code=code), "entry_point": "main"})
    return _EXPAND_PIPES[key]


class Pages:
    """One painted tile, sparse: page k (column k % per_row, row k // per_row) is a layer of `array` when something
    was painted there (`table[k]` = layer + 1), nothing otherwise (reads 0: nothing painted). The array grows as
    pages are added; the layers an undo empties are reused."""

    def __init__(self, device, size, fmt: str):
        self.device, self.format = device, supported(device, fmt)
        self.width, self.height = int(size[0]), int(size[1])
        self.side, self.per_row, self.rows = page_grid(size)
        self.table = np.zeros(self.per_row * self.rows, np.uint32)
        self.array = None
        self.layers = 0  # layers handed out (the free ones among them in `free`)
        self.free: list[int] = []
        self._buffer = None  # `table` on the GPU, made again when it changes

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def nbytes(self) -> int:
        return 0 if self.array is None else self.side * self.side * self.array.size[2] * DTYPE[self.format][1]

    def box(self, b: int) -> tuple[int, int, int, int]:
        """(x, y, w, h) of page `b` in the tile (the last row and column may be cut)."""
        x, y = (b % self.per_row) * self.side, (b // self.per_row) * self.side
        return x, y, min(self.side, self.width - x), min(self.side, self.height - y)

    def painted(self) -> list[int]:
        return [int(b) for b in np.nonzero(self.table)[0]]

    def buffer(self):
        """The table on the GPU (storage, u32 per page)."""
        if self._buffer is None:
            self._buffer = self.device.create_buffer_with_data(data=self.table.tobytes(),
                                                               usage=wgpu.BufferUsage.STORAGE)
        return self._buffer

    def view(self):
        """The array as a 2D array view (a 1-layer array with nothing painted yet)."""
        if self.array is None:
            self.array = _array(self.device, self.side, 1, self.format)
        return self.array.create_view(dimension="2d-array")

    def allocate(self, blocks) -> list[int]:
        """A layer for every page of `blocks` that has none, zeroed (nothing painted); submitted at once (the copy
        of a grown array goes before the caller's work). Returns the pages added."""
        new = [b for b in blocks if not self.table[b]]
        if not new:
            return new
        d = self.device
        reused = self.free[:len(new)]
        self.free = self.free[len(reused):]
        fresh = list(range(self.layers, self.layers + len(new) - len(reused)))
        self.layers += len(fresh)
        enc = d.create_command_encoder()
        old = None
        cap = 0 if self.array is None else self.array.size[2]
        if self.layers > cap:
            grown = _array(d, self.side, min(max(self.layers, 2 * cap, 4), len(self.table)), self.format)
            used = self.layers - len(fresh)
            if self.array is not None and used:
                enc.copy_texture_to_texture({"texture": self.array}, {"texture": grown}, (self.side, self.side, used))
            old, self.array = self.array, grown  # WebGPU creates it zeroed
        for layer in reused:  # an emptied layer: cleared
            v = self.array.create_view(dimension="2d", base_array_layer=layer, array_layer_count=1)
            enc.begin_render_pass(color_attachments=[{"view": v, "load_op": "clear", "store_op": "store",
                                                      "clear_value": (0, 0, 0, 0)}]).end()
        d.queue.submit([enc.finish()])
        if old is not None:
            old.destroy()
        for b, layer in zip(new, reused + fresh):
            self.table[b] = layer + 1
        self._buffer = None
        return new

    def release(self, blocks) -> None:
        """Pages emptied (nothing painted there): their layers are reused by the next pages added."""
        for b in blocks:
            if self.table[b]:
                self.free.append(int(self.table[b]) - 1)
                self.table[b] = 0
        self._buffer = None

    def expand(self, enc, dst, region=None) -> None:
        """Record the pages into `dst` (a storage texture of the tile's size), 0 where nothing is painted, over
        `region` (x0, y0, x1, y1; default all of it)."""
        x0, y0, x1, y1 = region or (0, 0, self.width, self.height)
        if x1 <= x0 or y1 <= y0:
            return
        d = self.device
        pipe = expand_pipe(d, self.format, dst.format)
        uni = struct.pack("8I", x0, y0, x1, y1, self.side, self.per_row, 0, 0)
        group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)}},
            {"binding": 1, "resource": self.view()},
            {"binding": 2, "resource": {"buffer": self.buffer()}},
            {"binding": 3, "resource": dst.create_view()}])
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups((x1 - x0 + 7) // 8, (y1 - y0 + 7) // 8)
        cp.end()

    def dense(self, usage=USAGE):
        """A whole RGBA texture of the pixels (the caller destroys it)."""
        t = self.device.create_texture(size=(self.width, self.height, 1), format=unpacked(self.format), usage=usage)
        enc = self.device.create_command_encoder()
        self.expand(enc, t)
        self.device.queue.submit([enc.finish()])
        return t

    def copy(self) -> Pages:
        c = Pages(self.device, self.size, self.format)
        c.copy_from(self)
        return c

    def copy_from(self, other: Pages) -> None:
        """Take `other`'s pixels (same size and format)."""
        assert (other.size, other.format) == (self.size, self.format)
        if self.array is not None:
            self.array.destroy()
        self.table, self.layers, self.free, self._buffer = other.table.copy(), other.layers, list(other.free), None
        self.array = None
        if other.array is not None and other.layers:
            self.array = _array(self.device, self.side, other.layers, self.format)
            enc = self.device.create_command_encoder()
            enc.copy_texture_to_texture({"texture": other.array}, {"texture": self.array},
                                        (self.side, self.side, other.layers))
            self.device.queue.submit([enc.finish()])

    def destroy(self) -> None:
        if self.array is not None:
            self.array.destroy()
        self.array = None

    # --- CPU access (save, load, structural changes) ------------------------------------------------

    def read(self) -> np.ndarray:
        """The pixels as float32 HxWx4 (0 where nothing is painted)."""
        out = np.zeros((self.height, self.width, 4), np.float32)
        dtype, bpp = DTYPE[self.format]
        comps = 2 if two(self.format) else 4
        for b in self.painted():
            x, y, w, h = self.box(b)
            row = (w * bpp + 255) // 256 * 256
            data = self.device.queue.read_texture({"texture": self.array, "origin": (0, 0, int(self.table[b]) - 1)},
                                                  {"bytes_per_row": row}, (w, h, 1))
            a = np.frombuffer(data, np.uint8).reshape(h, row)[:, :w * bpp].copy().view(dtype).reshape(h, w, comps)
            if comps == 2:  # value and how much is painted
                a = a[..., [0, 0, 0, 1]]
            out[y:y + h, x:x + w] = a.astype(np.float32) / 255.0 if dtype is np.uint8 else a
        return out

    def raw(self) -> tuple[np.ndarray, np.ndarray]:
        """(table, pages): what is painted as it is stored, for a file: per page its layer + 1 in `pages` (0: nothing
        painted), and the painted pages (n, side, side, components) in the storage's type."""
        dtype, bpp = DTYPE[self.format]
        comps = 2 if two(self.format) else 4
        used = self.painted()
        table = np.zeros_like(self.table)
        if not used:
            return table, np.zeros((0, self.side, self.side, comps), dtype)
        row = (self.side * bpp + 255) // 256 * 256
        data = self.device.queue.read_texture({"texture": self.array, "origin": (0, 0, 0)},
                                              {"bytes_per_row": row, "rows_per_image": self.side},
                                              (self.side, self.side, self.array.size[2]))
        layers = np.frombuffer(data, np.uint8).reshape(self.array.size[2], self.side, row)[:, :, :self.side * bpp]
        layers = layers.copy().view(dtype).reshape(-1, self.side, self.side, comps)
        for i, b in enumerate(used):
            table[b] = i + 1
        return table, np.ascontiguousarray(layers[[int(self.table[b]) - 1 for b in used]])

    def set_raw(self, table: np.ndarray, pages: np.ndarray) -> None:
        """Replace the pixels with `raw()`'s (the same size and storage)."""
        self.release(self.painted())
        if self.array is not None:
            self.array.destroy()
        self.table = np.asarray(table, np.uint32).copy()
        self.layers, self.free, self._buffer, self.array = len(pages), [], None, None
        if len(pages):
            self.array = _array(self.device, self.side, len(pages), self.format)
            bpp = DTYPE[self.format][1]
            self.device.queue.write_texture({"texture": self.array}, np.ascontiguousarray(pages).tobytes(),
                                            {"bytes_per_row": self.side * bpp, "rows_per_image": self.side},
                                            (self.side, self.side, len(pages)))

    def write(self, pixels: np.ndarray) -> None:
        """Replace the pixels (float HxWx4 of the tile's size): pages only where something is painted (alpha)."""
        assert pixels.shape[:2] == (self.height, self.width)
        self.release(self.painted())
        dtype, bpp = DTYPE[self.format]
        boxes = {b: self.box(b) for b in range(len(self.table))}
        used = [b for b, (x, y, w, h) in boxes.items() if np.any(pixels[y:y + h, x:x + w, 3] > 0)]
        self.allocate(used)
        for b in used:
            x, y, w, h = boxes[b]
            p = pixels[y:y + h, x:x + w]
            if two(self.format):
                p = p[..., [0, 3]]
            p = np.clip(p, 0, 1) * 255.0 + 0.5 if dtype is np.uint8 else p
            data = np.ascontiguousarray(p.astype(dtype))
            self.device.queue.write_texture({"texture": self.array, "origin": (0, 0, int(self.table[b]) - 1)},
                                            data.tobytes(), {"bytes_per_row": w * bpp}, (w, h, 1))


def _whole(grid: Pages, storage: str, table, pages) -> np.ndarray:
    """A file's pages (`Pages.raw`, of `storage`) as whole float RGBA pixels."""
    out = np.zeros((grid.height, grid.width, 4), np.float32)
    for b in np.nonzero(table)[0]:
        x, y, w, h = grid.box(int(b))
        a = pages[int(table[b]) - 1][:h, :w]
        a = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)
        out[y:y + h, x:x + w] = a[..., [0, 0, 0, 1]] if two(storage) else a
    return out


@dataclass
class Entry:
    pages: Pages
    revision: int = 0


class PaintStore:
    def __init__(self, device):
        self.device = device
        self.entries: dict[tuple[str, str, int], Entry] = {}  # (uid, channel or "mask", UDIM tile) -> entry

    def _bump(self, e: Entry) -> None:
        """A new revision, unique in the process: stores of reopened projects (same uids) share the evaluator's
        cache, a revision counted per store would find a previous one's pixels."""
        e.revision = next(_REVISIONS)

    def get(self, uid: str, key: str, udim: int) -> Entry | None:
        return self.entries.get((uid, key, udim))

    def ensure(self, uid: str, key: str, udim: int, size, storage: str) -> Entry:
        """The entry, created empty (nothing painted) or resampled to `size` / `storage` if it differs."""
        e = self.entries.get((uid, key, udim))
        size = (int(size[0]), int(size[1]))
        storage = supported(self.device, storage)
        if e is not None and (e.pages.size, e.pages.format) == (size, storage):
            return e
        p = Pages(self.device, size, storage)
        if e is None:
            e = self.entries[(uid, key, udim)] = Entry(p)
        else:
            p.write(_resample(e.pages.read(), *size))
            e.pages.destroy()
            e.pages = p
        self._bump(e)
        return e

    def touched(self, uid: str, key: str, udim: int) -> None:
        """Call after writing an entry's pages on the GPU."""
        self._bump(self.entries[(uid, key, udim)])

    def read(self, uid: str, key: str, udim: int) -> np.ndarray | None:
        e = self.get(uid, key, udim)
        return None if e is None else e.pages.read()

    def write(self, uid: str, key: str, udim: int, pixels: np.ndarray, storage: str) -> None:
        h, w = pixels.shape[:2]
        e = self.ensure(uid, key, udim, (w, h), storage)
        e.pages.write(pixels)
        self._bump(e)

    def write_raw(self, uid: str, key: str, udim: int, size, storage: str, table, pages, stored: str) -> None:
        """An entry of `storage` from a file's pages (`Pages.raw`, saved as `stored`): laid as they are when this
        device keeps them so, else through the whole pixels (another format: a device without two channel formats,
        or the entry's storage changed since the file was saved)."""
        e = self.ensure(uid, key, udim, size, storage)
        if e.pages.format == stored and len(table) == len(e.pages.table):
            e.pages.set_raw(table, pages)
        else:
            grid = Pages.__new__(Pages)  # the file's page grid
            grid.width, grid.height = int(size[0]), int(size[1])
            grid.side, grid.per_row, grid.rows = page_grid(size)
            e.pages.write(_whole(grid, stored, table, pages))
        self._bump(e)

    def nbytes(self) -> int:
        return sum(e.pages.nbytes for e in self.entries.values())

    # --- copies (undo of structural changes, duplicated layers) -------------------------------------

    def snapshot(self, keys) -> dict:
        """GPU copies of entries ({(uid, key, tile): Pages or None if absent}) to restore later."""
        return {k: (self.entries[k].pages.copy() if k in self.entries else None) for k in keys}

    def restore(self, snap: dict) -> None:
        for k, p in snap.items():
            old = self.entries.pop(k, None)
            if p is not None:
                e = self.entries[k] = Entry(p.copy())
                self._bump(e)
            if old is not None:
                old.pages.destroy()

    # --- undo of strokes: the pages a stroke reached, kept on the GPU -------------------------------

    def stroke_undo(self, key: tuple, saved) -> dict | None:
        """What undoing / redoing a stroke on entry `key` needs, on the GPU (no read back: releasing the mouse never
        waits): the pages it changed, before (brush.Saved: their saved layer, or nothing painted) and after (copies
        of the pages now). None when the stroke changed nothing."""
        p = self.entries[key].pages
        layers = {b: s for b, s in saved.layers.get(key, {}).items() if s is not None or p.table[b]}
        if not layers:
            return None
        blocks = sorted(layers)
        n = len(blocks)
        before, after = _array(self.device, p.side, n, p.format), _array(self.device, p.side, n, p.format)
        enc = self.device.create_command_encoder()
        for i, b in enumerate(blocks):
            one = (p.side, p.side, 1)
            if layers[b] is not None:
                enc.copy_texture_to_texture({"texture": saved.arrays[key], "origin": (0, 0, layers[b])},
                                            {"texture": before, "origin": (0, 0, i)}, one)
            if p.table[b]:
                enc.copy_texture_to_texture({"texture": p.array, "origin": (0, 0, int(p.table[b]) - 1)},
                                            {"texture": after, "origin": (0, 0, i)}, one)
        self.device.queue.submit([enc.finish()])
        return {"blocks": blocks, "before": before, "after": after,
                "empty": {"before": {b for b in blocks if layers[b] is None},
                          "after": {b for b in blocks if not p.table[b]}}}

    def apply_undo(self, key: tuple, snap: dict, after: bool) -> None:
        """Put a stroke_undo's before (or after) pages back into the entry."""
        e = self.entries.get(key)
        if e is None:
            return
        p, side = e.pages, e.pages.side
        which = "after" if after else "before"
        empty = snap["empty"][which]
        p.release([b for b in snap["blocks"] if b in empty])
        p.allocate([b for b in snap["blocks"] if b not in empty])
        enc = self.device.create_command_encoder()
        for i, b in enumerate(snap["blocks"]):
            if b not in empty:
                enc.copy_texture_to_texture({"texture": snap[which], "origin": (0, 0, i)},
                                            {"texture": p.array, "origin": (0, 0, int(p.table[b]) - 1)},
                                            (side, side, 1))
        self.device.queue.submit([enc.finish()])
        self._bump(e)

    def duplicate(self, uid: str, new_uid: str) -> None:
        for (u, key, udim), e in list(self.entries.items()):
            if u == uid:
                n = self.entries[(new_uid, key, udim)] = Entry(e.pages.copy())
                self._bump(n)

    def keys_of(self, uid: str) -> list[tuple[str, str, int]]:
        return [k for k in self.entries if k[0] == uid]

    def drop_unused(self, uids: set) -> None:
        for k in [k for k in self.entries if k[0] not in uids]:
            self.entries.pop(k).pages.destroy()


def _resample(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinear resampling on the CPU (only when a Texture Set changes size)."""
    sh, sw = img.shape[:2]
    if (sw, sh) == (w, h):
        return img
    ys = (np.arange(h) + 0.5) * sh / h - 0.5
    xs = (np.arange(w) + 0.5) * sw / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, sh - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, sw - 1)
    y1, x1 = np.clip(y0 + 1, 0, sh - 1), np.clip(x0 + 1, 0, sw - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None, None]
    fx = np.clip(xs - x0, 0, 1)[None, :, None]
    a = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    b = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return (a * (1 - fy) + b * fy).astype(np.float32)
