"""The brush: strokes of stamps painted in texture space (Paint tool and Eraser, paint-brush page).

A stroke starts with the camera buffer: the visible surface under every viewport pixel (world position,
normal), rendered once since the camera does not move while painting and read back to place the stamps. The
pointer path is walked pixel by pixel; a stamp is placed every Spacing (% of the stamp size) of surface
distance on the selected Texture Set. Stamps accumulate into the stroke's coverage texture on the GPU
(brush.wgsl), which is then laid over the painted entries as they were before the stroke (commit.wgsl), so
Stroke Opacity caps the whole stroke.

Ours where the documentation gives no number: Size is a percentage of half the mesh's bounding box diagonal
(Object size space) or of the viewport height (Viewport); the defaults; the jitter ranges (Angle Jitter 1 =
±180°, Position Jitter 1 = one stamp size).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import wgpu

from trafilatura3d.nodes.base import INT_MATH_WGSL, two_channels
from trafilatura3d.paint.model import local_uv, tile_uv
from trafilatura3d.paint.paint import page_grid, two, unpacked
from trafilatura3d.paint.uvmaps import UVMaps

WGSL = __import__("pathlib").Path(__file__).parent / "wgsl"
# The projection of the Projection tool and the stencil (project.wgsl here, the viewport's overlays in render3d)
PROJECTION_WGSL = INT_MATH_WGSL + (WGSL / "projection.wgsl").read_text(encoding="utf-8")
ALIGNMENTS = ("Camera", "Tangent | Wrap", "Tangent | Planar")  # UV alignment not offered yet
SIZE_SPACES = ("Object", "Viewport")  # Texture size space not offered yet (no painting in the 2D view)
MAX_STAMPS = 4096
STAMP_BYTES = 64


@dataclass
class BrushSettings:
    """The Brush section of the Paint tool's properties (contextual toolbar: Size, Flow, Stroke opacity,
    Spacing)."""
    size: float = 20.0
    flow: float = 100.0
    stroke_opacity: float = 100.0
    spacing: float = 10.0
    angle: float = 0.0  # degrees
    follow_path: bool = False
    size_jitter: float = 0.0
    flow_jitter: float = 0.0
    angle_jitter: float = 0.0
    position_jitter: float = 0.0
    alignment: str = "Tangent | Wrap"  # the documentation's default
    backface_culling: float = 90.0  # degrees
    size_space: str = "Object"  # the documentation's default
    size_pressure: bool = False
    flow_pressure: bool = False
    stamps_blending: str = "Normal"  # "Lighten": overlapping stamps keep the strongest instead of building up


from trafilatura3d.paint.alphas import ALPHAS, DEFAULT_ALPHA  # noqa: E402,F401 - the brush's stamp shapes


@dataclass
class View:
    """The camera the stroke is painted from."""
    view_proj: np.ndarray  # 4x4, column vectors (render3d convention)
    eye: np.ndarray
    fov: float  # vertical, radians
    width: int  # camera buffer pixels
    height: int


@dataclass
class Target:
    """A painted entry the stroke writes: its pages (paint.Pages) and the value laid (RGB), or erase. `source`: the
    values per texel instead (rgba32float in texture space, alpha = how much is laid), e.g. the Projection tool's.
    `color`, `hdr`: the channel's (linear blending, no clamp) when the stroke blends (`Stroke.blending`)."""
    key: tuple
    pages: object
    value: tuple = (1.0, 1.0, 1.0)
    erase: bool = False
    source: object = None
    color: bool = False
    hdr: bool = False


TILING_MODES = ("No Tiling", "Horizontal Tiling", "Vertical Tiling", "H and V Tiling")  # the stencil's (Paint page)
TEXTURE_ALPHA = "texture:"  # an alpha naming a project texture (its luminance)
SYMMETRY_TYPES = ("Mirror", "Radial")
FLIP_COPIES = ("None", "U", "V")  # Radial symmetry's Flip copy (ours: the page names U or V, not the list)
FLIP_CODE = 1000.0  # a stamp's angle + FLIP_CODE x (1: flip U, 2: flip V), decoded by brush.wgsl


def symmetry_copies(kind: str, axis: int, position, copies: int = 2, flip: str = "None") -> list:
    """The copies a symmetry adds to every stamp, (matrix, offset, reflection, flip): Mirror reflects across the
    plane `axis` = position[axis]; Radial turns around the line along `axis` through `position` in `copies`
    equal steps, Flip copy mirroring every other copy in the stamp's U or V (Symmetry pages; the number of copies
    is ours, the pages do not list it). Scene space."""
    pos = np.asarray(position, np.float64) if np.ndim(position) else np.eye(3)[axis] * float(position)
    if kind == "Mirror":
        m = np.diag([-1.0 if i == axis else 1.0 for i in range(3)])
        return [(m, pos - m @ pos, True, 0)]
    out = []
    a = np.eye(3)[axis]
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    for i in range(1, max(int(copies), 1)):
        th = 2 * math.pi * i / copies
        m = np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k
        out.append((m, pos - m @ pos, False, FLIP_COPIES.index(flip) if i % 2 else 0))
    return out
PROJECTION_MODES = ("Projection",)  # Physical projection (particles) is not offered


@dataclass
class ProjectionSettings:
    """The Projection tool's (and the stencil's) image in the viewport: transform (S + left drag rotates, S + right
    drag scales, S + middle drag moves; offset in fractions of the viewport, rotation in degrees, scale 1 = the
    image's height fills the viewport's), flips, tiling, the preview's opacity, and the projection's falloff as in
    Mari's projection settings: faces turned away from the view by more than `fade_start` degrees fade until
    `fade_end`, hidden surfaces are not reached (`occlusion`), the falloff shown in red over the view
    (`show_falloff`). Ours: the documentation gives the tool's controls, not these."""
    offset: tuple = (0.0, 0.0)
    rotation: float = 0.0
    scale: float = 1.0
    tiling: str = "H and V Tiling"
    mode: str = "Projection"
    flip_h: bool = False
    flip_v: bool = False
    opacity: float = 0.5
    fade_start: float = 60.0
    fade_end: float = 85.0
    occlusion: bool = True
    show_falloff: bool = True

    def xf(self) -> tuple:
        return (*self.offset, math.radians(self.rotation), self.scale)

    def tiling_flags(self) -> int:
        """The tiling (0-3) with the flips (bits 2 and 3), as projection.wgsl reads them."""
        return TILING_MODES.index(self.tiling) | int(self.flip_h) << 2 | int(self.flip_v) << 3

    def fade(self) -> tuple:
        """(cos of the start angle, cos of the end angle): the weight's range on the facing ratio."""
        return (math.cos(math.radians(min(self.fade_start, self.fade_end - 1e-3))),
                math.cos(math.radians(self.fade_end)))


# The camera buffer: per pixel the world position (A = material + 1), the normal (half float) and the UV (the
# Clone and Smudge tools read the texel under any pixel). 32 bytes per sample, the attachments' limit.
CAMERA_SHADER = """
struct U { view_proj: mat4x4f };
@group(0) @binding(0) var<uniform> u: U;
struct VOut { @builtin(position) pos: vec4f, @location(0) world: vec3f, @location(1) normal: vec3f,
              @location(2) @interpolate(flat) material: f32, @location(3) uv: vec2f };
@vertex
fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) uv: vec2f,
      @builtin(instance_index) m: u32) -> VOut {
    var o: VOut;
    o.pos = u.view_proj * vec4f(p, 1.0);
    o.world = p; o.normal = n; o.material = f32(m); o.uv = uv;
    return o;
}
struct F { @location(0) a: vec4f, @location(1) b: vec4f, @location(2) c: vec2f };
@fragment
fn fs(i: VOut, @builtin(front_facing) front: bool) -> F {
    var n = normalize(i.normal);
    if (!front) { n = -n; }
    return F(vec4f(i.world, i.material + 1.0), vec4f(n, 1.0), i.uv);
}
"""


# Polygon Fill: triangles rasterized in texture space as full coverage.
FILL_SHADER = """
@vertex
fn vs(@location(0) uv: vec2f) -> @builtin(position) vec4f { return vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0); }
@fragment
fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
"""


EMPTY_RECT = struct.pack("4I", 0xFFFFFFFF, 0xFFFFFFFF, 0, 0)  # min x, min y, max x, max y of nothing


class Saved:
    """What a stroke's targets held before it, per target key: the pages it changed, each saved the first time (a
    layer of a texture array per page, grown as needed), or None for a page nothing was painted on."""

    def __init__(self, device):
        self.device = device
        self.arrays, self.layers = {}, {}
        self._old = []  # arrays outgrown during the stroke (in use by submitted copies until the end)
        self._placeholders = {}  # format -> 1x1 array bound while a target has nothing saved

    def keys(self) -> list:
        return list(self.layers)

    def save(self, enc, t, blocks, added) -> None:
        """Copy the pages of target `t` in `blocks` not saved yet; those in `added` had nothing painted."""
        layers = self.layers.setdefault(t.key, {})
        new = [b for b in blocks if b not in layers]
        kept = [b for b in new if b not in added and t.pages.table[b]]
        for b in new:
            layers[b] = None
        if not kept:
            return
        p = t.pages
        arr = self.arrays.get(t.key)
        count = sum(v is not None for v in layers.values())
        need = count + len(kept)
        if arr is None or arr.size[2] < need:
            cap = min(max(need, 2 * (arr.size[2] if arr else 4)), len(p.table))
            grown = self.device.create_texture(size=(p.side, p.side, cap), format=p.format,
                                               usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST
                                               | wgpu.TextureUsage.COPY_SRC)
            if arr is not None and count:
                enc.copy_texture_to_texture({"texture": arr}, {"texture": grown}, (p.side, p.side, count))
            if arr is not None:
                self._old.append(arr)
            arr = self.arrays[t.key] = grown
        for b in kept:
            layers[b] = count
            enc.copy_texture_to_texture({"texture": p.array, "origin": (0, 0, int(p.table[b]) - 1)},
                                        {"texture": arr, "origin": (0, 0, count)}, (p.side, p.side, 1))
            count += 1

    def of(self, key, pages):
        """(texture array, table buffer: per page (0 not changed, 1 nothing painted before, saved layer + 2;
        the page's layer + 1)) of a target, for commit.wgsl."""
        table = np.zeros((len(pages.table), 2), np.uint32)
        for b, layer in self.layers.get(key, {}).items():
            table[b, 0] = 1 if layer is None else layer + 2
        table[:, 1] = pages.table
        arr = self.arrays.get(key)
        if arr is None:  # nothing saved yet: a bound placeholder (not kept as the saved array: save grows that)
            arr = self._placeholders.get(pages.format)
            if arr is None:
                arr = self._placeholders[pages.format] = self.device.create_texture(
                    size=(1, 1, 1), format=pages.format, usage=wgpu.TextureUsage.TEXTURE_BINDING)
        return arr, self.device.create_buffer_with_data(data=table.tobytes(), usage=wgpu.BufferUsage.STORAGE)

    def destroy(self) -> None:
        for t in [*self.arrays.values(), *self._old, *self._placeholders.values()]:
            t.destroy()
        self.arrays, self.layers, self._old, self._placeholders = {}, {}, [], {}


class BrushEngine:
    def __init__(self, device):
        self.device = device
        d = device
        self.brush_pipe = d.create_compute_pipeline(layout="auto", compute={
            "module": d.create_shader_module(code=(WGSL / "brush.wgsl").read_text(encoding="utf-8")), "entry_point": "main"})
        self._commit = {}
        mod = d.create_shader_module(code=CAMERA_SHADER)
        vb = [{"array_stride": s, "attributes": [{"shader_location": i, "offset": 0, "format": f}]}
              for i, (s, f) in enumerate([(12, "float32x3"), (12, "float32x3"), (8, "float32x2")])]
        self.camera_pipe = d.create_render_pipeline(
            layout="auto", vertex={"module": mod, "entry_point": "vs", "buffers": vb},
            fragment={"module": mod, "entry_point": "fs",
                      "targets": [{"format": "rgba32float"}, {"format": "rgba16float"}, {"format": "rg32float"}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth24plus", "depth_write_enabled": True, "depth_compare": "less"})
        self.sampler = d.create_sampler(mag_filter="linear", min_filter="linear")
        code = (WGSL / "project.wgsl").read_text(encoding="utf-8").replace("//INCLUDE projection.wgsl",
                                                                           PROJECTION_WGSL)
        self.project_pipe = d.create_compute_pipeline(layout="auto", compute={
            "module": d.create_shader_module(code=code), "entry_point": "main"})
        self.no_source = d.create_texture(size=(1, 1, 1), format="rgba32float",
                                          usage=wgpu.TextureUsage.TEXTURE_BINDING)
        self.no_uv = d.create_texture(size=(1, 1, 1), format="rg32float", usage=wgpu.TextureUsage.TEXTURE_BINDING)
        self.resample_pipe = d.create_compute_pipeline(layout="auto", compute={
            "module": d.create_shader_module(code=(WGSL / "resample.wgsl").read_text(encoding="utf-8")), "entry_point": "main"})
        fill = d.create_shader_module(code=FILL_SHADER)
        self.fill_pipe = d.create_render_pipeline(
            layout="auto", vertex={"module": fill, "entry_point": "vs", "buffers": [
                {"array_stride": 8, "attributes": [{"shader_location": 0, "offset": 0, "format": "float32x2"}]}]},
            fragment={"module": fill, "entry_point": "fs", "targets": [{"format": "r32float"}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"})
        self._alphas = {}

    def commit_pipe(self, fmt):
        """commit.wgsl for an entry storage format."""
        if fmt not in self._commit:
            code = (WGSL / "blend_lib.wgsl").read_text(encoding="utf-8") + (WGSL / "commit.wgsl").read_text(
                encoding="utf-8").replace("FORMAT", fmt).replace("TWO_ENTRY", str(two(fmt)).lower())
            self._commit[fmt] = self.device.create_compute_pipeline(layout="auto", compute={
                "module": self.device.create_shader_module(code=code), "entry_point": "main"})
        return self._commit[fmt]

    def alpha(self, name: str, image: np.ndarray | None = None):
        """The stamp texture of a built-in alpha (or of `image`, grayscale HxW in 0-1)."""
        if name not in self._alphas:
            a = ALPHAS[name](256) if image is None else image
            a = np.clip(np.asarray(a, np.float32), 0, 1)
            h, w = a.shape
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., 0] = (a * 255 + 0.5).astype(np.uint8)
            rgba[..., 3] = 255
            t = self.device.create_texture(size=(w, h, 1), format="rgba8unorm",
                                           usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
            self.device.queue.write_texture({"texture": t}, rgba.tobytes(), {"bytes_per_row": w * 4}, (w, h, 1))
            self._alphas[name] = t
        return self._alphas[name]

    def project(self, maps: UVMaps, view: View, settings: ProjectionSettings, color, image=None,
                single_channel=False, gray=False, camera=None, falloff=False, diag=1.0):
        """The Projection tool's (or the stencil's) values in texture space (rgba32float, project.wgsl): `image` (a
        texture, or None for the uniform `color`) seen through the projection from `view`; `falloff`: the
        settings' fade by angle and occlusion (the latter needs the view's `camera` buffer)."""
        d = self.device
        w, h = maps.size
        U = wgpu.TextureUsage
        dst = d.create_texture(size=(w, h, 1), format="rgba32float", usage=U.STORAGE_BINDING | U.TEXTURE_BINDING)
        vp = np.identity(4, np.float32) if view is None else np.asarray(view.view_proj, np.float32)
        aspect = 1.0 if view is None else view.width / max(view.height, 1)
        eye = (0.0, 0.0, 0.0) if view is None else tuple(float(x) for x in view.eye)
        occlusion = falloff and settings.occlusion and camera is not None
        size = (view.width, view.height) if view is not None else (1, 1)
        uni = np.ascontiguousarray(vp.T).tobytes() + struct.pack(
            "24f", *settings.xf(), aspect, float(settings.tiling_flags()),
            1.0 if image is not None else 0.0, 1.0 if gray else 0.0, *color[:3], 1.0 if single_channel else 0.0,
            *eye, 1.0 if occlusion else 0.0, *settings.fade(), 1.0 if falloff else 0.0, diag * 2e-3,
            float(size[0]), float(size[1]), 0.0, 0.0)
        group = d.create_bind_group(layout=self.project_pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)}},
            {"binding": 1, "resource": maps.position.create_view()},
            {"binding": 2, "resource": (image or self.no_source).create_view()},
            {"binding": 3, "resource": dst.create_view()},
            {"binding": 4, "resource": maps.normal.create_view()},
            {"binding": 5, "resource": (camera.texture if camera is not None else maps.position).create_view()}])
        enc = d.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.project_pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        cp.end()
        d.queue.submit([enc.finish()])
        return dst

    def resample(self, maps: UVMaps, view: View, camera: CameraBuffer, entry, below, delta, material: int,
                 diag: float, udim: int, dst=None):
        """The content seen `delta` camera buffer pixels away from every visible texel (resample.wgsl): the
        Clone tool's source, a Smudge stamp's. `entry` the painted texture, `below` the stack below (Passthrough)
        or None. Into `dst` (rgba32float, made when None). `udim`: the tile of `maps` and `entry` (the camera's UVs
        moved into its square; a source in another tile gives nothing)."""
        d = self.device
        w, h = maps.size
        U = wgpu.TextureUsage
        if dst is None:
            dst = d.create_texture(size=(w, h, 1), format="rgba32float", usage=U.STORAGE_BINDING | U.TEXTURE_BINDING)
        if view is None:  # texture space: `delta` in UV (Smudge along path)
            vp, size, eye, uv_mode = np.identity(4, np.float32), (1.0, 1.0), (0.0, 0.0, 0.0), 1.0
        else:
            vp, size, eye, uv_mode = np.asarray(view.view_proj, np.float32), (view.width, view.height), view.eye, 0.0
        du, dv = tile_uv(udim)
        uni = np.ascontiguousarray(vp.T).tobytes() + struct.pack(
            "16f", float(delta[0]), float(delta[1]), float(material + 1), 1.0 if below is not None else 0.0,
            float(size[0]), float(size[1]), diag * 2e-3, uv_mode, *eye, 1.0, -du, dv,
            1.0 if below is not None and two_channels(below.format) else 0.0, 0.0)
        cam_pos = camera.texture if camera is not None else maps.position
        cam_uv = camera.uvs if camera is not None else self.no_uv
        group = d.create_bind_group(layout=self.resample_pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)}},
            {"binding": 1, "resource": maps.position.create_view()},
            {"binding": 2, "resource": cam_pos.create_view()},
            {"binding": 3, "resource": cam_uv.create_view()},
            {"binding": 4, "resource": entry.create_view()},
            {"binding": 5, "resource": (below or entry).create_view()},
            {"binding": 6, "resource": dst.create_view()}])
        enc = d.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.resample_pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        cp.end()
        d.queue.submit([enc.finish()])
        return dst

    def camera_buffer(self, meshes: list, view: View) -> CameraBuffer:
        """The visible surface per pixel of the view (CameraBuffer). `meshes`: (vertex buffers [position, normal,
        …], index buffer, submeshes [(start, count, material)])."""
        d = self.device
        w, h = view.width, view.height
        U = wgpu.TextureUsage
        pos = d.create_texture(size=(w, h, 1), format="rgba32float", usage=U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC)
        nrm = d.create_texture(size=(w, h, 1), format="rgba16float",
                               usage=U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC)
        uv = d.create_texture(size=(w, h, 1), format="rg32float", usage=U.RENDER_ATTACHMENT | U.TEXTURE_BINDING | U.COPY_SRC)
        depth = d.create_texture(size=(w, h, 1), format="depth24plus", usage=U.RENDER_ATTACHMENT)
        ub = d.create_buffer_with_data(data=np.ascontiguousarray(view.view_proj.T, np.float32).tobytes(),
                                       usage=wgpu.BufferUsage.UNIFORM)
        group = d.create_bind_group(layout=self.camera_pipe.get_bind_group_layout(0),
                                    entries=[{"binding": 0, "resource": {"buffer": ub}}])
        enc = d.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[{"view": t.create_view(), "load_op": "clear", "store_op": "store",
                                "clear_value": (0, 0, 0, 0)} for t in (pos, nrm, uv)],
            depth_stencil_attachment={"view": depth.create_view(), "depth_load_op": "clear",
                                      "depth_store_op": "discard", "depth_clear_value": 1.0})
        rp.set_pipeline(self.camera_pipe)
        rp.set_bind_group(0, group)
        for vbs, ib, subs in meshes:
            rp.set_vertex_buffer(0, vbs[0])
            rp.set_vertex_buffer(1, vbs[1])
            rp.set_vertex_buffer(2, vbs[3])  # [position, normal, tangent, uv]
            rp.set_index_buffer(ib, "uint32")
            for start, count, material in subs:
                rp.draw_indexed(count, 1, start, 0, material)
        rp.end()
        d.queue.submit([enc.finish()])
        return CameraBuffer(d, pos, nrm, uv)


class CameraBuffer:
    """The visible surface under every pixel of a view, on the GPU (position with A = material + 1, 0 = nothing;
    normal). The brush and the cursor read it around the pointer only: blocks of BLOCK pixels are read back the
    first time they are needed (a stroke crosses a few; reading the whole buffer took ~90 ms)."""
    BLOCK = 64
    WHOLE_READ = 8  # `sees` needing more blocks than this reads the positions whole, once

    def __init__(self, device, positions, normals, uvs=None):
        self.device, self.texture, self.normals, self.uvs = device, positions, normals, uvs
        self.width, self.height = positions.width, positions.height
        self.blocks: dict = {}
        self._positions = None  # all of them, read by `sees` for a large area

    def _block(self, bx, by):
        if (bx, by) not in self.blocks:
            B = self.BLOCK
            x, y = bx * B, by * B
            w, h = min(B, self.width - x), min(B, self.height - y)
            self.blocks[(bx, by)] = (_read(self.device, self.texture, (x, y, w, h)),
                                     _read(self.device, self.normals, (x, y, w, h)))
        return self.blocks[(bx, by)]

    def hit(self, x: float, y: float):
        """(position xyz + material + 1, normal) under a pixel, or None outside the view."""
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.width and 0 <= yi < self.height):
            return None
        pos, nrm = self._block(xi // self.BLOCK, yi // self.BLOCK)
        return pos[yi % self.BLOCK, xi % self.BLOCK], nrm[yi % self.BLOCK, xi % self.BLOCK, :3]

    def sees(self, view: View, points, material: int, tol: float) -> np.ndarray:
        """Per world point ((M, 3)): is it the visible surface of `material` under its pixel? Within `tol` plus two
        pixels' footprint at its distance."""
        p = np.asarray(points, np.float64)
        out = np.zeros(len(p), bool)
        if not len(p):
            return out
        clip = np.concatenate([p, np.ones((len(p), 1))], 1) @ np.asarray(view.view_proj, np.float64).T
        w = np.where(clip[:, 3] > 1e-9, clip[:, 3], 1.0)
        xi = np.floor((clip[:, 0] / w * 0.5 + 0.5) * self.width).astype(np.int64)
        yi = np.floor((0.5 - clip[:, 1] / w * 0.5) * self.height).astype(np.int64)
        idx = np.where((clip[:, 3] > 1e-9) & (xi >= 0) & (xi < self.width) & (yi >= 0) & (yi < self.height))[0]
        B = self.BLOCK
        block = (yi[idx] // B) * (self.width // B + 1) + xi[idx] // B
        if self._positions is None and len(np.unique(block)) > self.WHOLE_READ:  # one read beats many blocks
            self._positions = _read(self.device, self.texture)
        if self._positions is not None:
            q = self._positions[yi[idx], xi[idx]]
        else:
            order = np.argsort(block, kind="stable")
            idx, block = idx[order], block[order]
            starts = np.concatenate([[0], np.nonzero(np.diff(block))[0] + 1, [len(idx)]])
            q = np.empty((len(idx), 4), np.float32)
            for a, b in zip(starts[:-1], starts[1:]):  # one read per block of pixels (kept for the next ones)
                k = idx[a:b]
                pos, _ = self._block(int(xi[k[0]] // B), int(yi[k[0]] // B))
                q[a:b] = pos[yi[k] % B, xi[k] % B]
        dist = np.linalg.norm(p[idx] - np.asarray(view.eye, np.float64), axis=1)
        foot = dist * 2 * math.tan(view.fov / 2) / max(view.height, 1)
        out[idx] = (np.round(q[:, 3]) - 1 == material) & (np.linalg.norm(q[:, :3] - p[idx], axis=1) <= tol + 2 * foot)
        return out


def stamp_radius(s: BrushSettings, view: View, hit, diag: float, pressure: float = 1.0) -> tuple[float, float]:
    """(world radius, pixel radius) of a stamp at `hit` (also the viewport's brush cursor)."""
    size = s.size * (pressure if s.size_pressure else 1.0)
    dist = float(np.linalg.norm(np.asarray(hit) - view.eye))
    px_per_world = (view.height / 2) / max(dist * math.tan(view.fov / 2), 1e-9)
    if s.size_space == "Viewport":
        rpx = size / 100 * view.height / 2 / 2
        return rpx / px_per_world, rpx
    rw = size / 100 * diag / 2 / 2
    return rw, rw * px_per_world


_FLOAT_FORMATS = {"rgba32float": (np.float32, 4), "rgba16float": (np.float16, 4), "rg32float": (np.float32, 2)}


def _read(device, t, region=None) -> np.ndarray:
    """A float texture (or a region (x, y, w, h) of it) as HxWxC float32."""
    dtype, ch = _FLOAT_FORMATS[t.format]
    bpp = np.dtype(dtype).itemsize * ch
    x, y, w, h = region or (0, 0, t.width, t.height)
    row = (w * bpp + 255) // 256 * 256
    data = device.queue.read_texture({"texture": t, "origin": (x, y, 0)}, {"bytes_per_row": row}, (w, h, 1))
    a = np.frombuffer(data, np.uint8).reshape(h, row)[:, :w * bpp]
    return a.copy().view(dtype).reshape(h, w, ch).astype(np.float32)


@dataclass
class TileTargets:
    """What a stroke writes in one UDIM tile (each tile is its own texture): the tile's mesh in texture space
    (`maps`), its painted entries, the quick mask protecting them (`limit`, R protects) and the stencil in its
    texture space (grayscale), or None."""
    udim: int
    maps: UVMaps
    targets: list
    limit: object = None
    stencil: object = None


class Stroke:
    """One stroke: `add` pointer positions (camera buffer pixels, with pressure), `end` when released.
    Painting only lands on the Texture Set whose mesh material is `material`. The stamps are placed once, on the
    surface, and laid in every tile they reach (`tiles`: TileTargets): a stroke paints across the tiles."""

    def __init__(self, engine: BrushEngine, tiles: list[TileTargets], view: View, meshes: list, material: int,
                 settings: BrushSettings, bbox_diagonal: float, alpha: str = DEFAULT_ALPHA, seed: int = 0,
                 camera=None, symmetry=None, smudge=None, blending: tuple[str, float] | None = None):
        self.engine, self.tiles, self.view = engine, tiles, view
        # (blending mode identifier, opacity): the value is blended with what the entries hold in that mode, as a
        # layer over the stack (Polygon Fill's Blending), instead of laid over it at the Stroke opacity
        self.blending = blending
        # Smudge tool: {target key: the stack below (Passthrough) or None}; every stamp drags the content from
        # where the previous one was.
        self.smudge = smudge
        self._smudge_prev = None
        self.material, self.s, self.diag = material, settings, bbox_diagonal
        self.alpha_tex = engine.alpha(alpha)
        self.rng = np.random.default_rng(seed)
        self.symmetry = symmetry  # the copies of every stamp: `symmetry_copies`, None = off
        d = engine.device
        # The camera buffer of this view (`camera`: one already rendered, e.g. the viewport's hover buffer).
        # Without a view (painting in the 2D view, add_texture) there is none: the shader does not read it.
        self.camera = None if view is None else (camera or engine.camera_buffer(meshes, view))
        U = wgpu.TextureUsage
        self.stroke_texs = []  # per tile: the stroke's coverage
        for tile in tiles:
            w, h = tile.maps.size
            t = d.create_texture(size=(w, h, 1), format="r32float",
                                 usage=U.STORAGE_BINDING | U.TEXTURE_BINDING | U.COPY_DST | U.COPY_SRC
                                 | U.RENDER_ATTACHMENT)
            self.stroke_texs.append(t)
        enc = d.create_command_encoder()  # cleared on the GPU (no zeros sent from the CPU)
        for t in self.stroke_texs:
            enc.begin_render_pass(color_attachments=[{"view": t.create_view(), "load_op": "clear",
                                                      "store_op": "store", "clear_value": (0, 0, 0, 0)}]).end()
        d.queue.submit([enc.finish()])
        # per tile the texels written since `take_dirty` (brush.wgsl), and whether a pass wrote all of it (fill)
        self.dirty = [d.create_buffer_with_data(data=EMPTY_RECT, usage=wgpu.BufferUsage.STORAGE
                                                | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
                      for _ in tiles]
        self._whole = [False] * len(tiles)
        # What the entries held before the stroke (the commit lays the stroke over it, the undo puts it back): the
        # pages it changes, saved the first time (a stroke reaching a little of a 4096 texture copies a little of
        # it).
        self.saved = Saved(d)
        self.blocks = []  # per tile: (page side, pages per row, the bits buffer the brush sets)
        for tile in tiles:
            size, per_row, rows = page_grid(tile.maps.size)
            words = -(-(per_row * rows) // 32)
            self.blocks.append((size, per_row, d.create_buffer_with_data(
                data=bytes(4 * words), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST)))
        self.base = None  # Smudge: the entries right before each stamp, whole
        if smudge is not None:
            self.base = {t.key: d.create_texture(size=(*t.pages.size, 1), format=unpacked(t.pages.format),
                                                 usage=U.TEXTURE_BINDING | U.STORAGE_BINDING) for t in self.targets()}
            for tile in tiles:
                for t in tile.targets:
                    t.source = d.create_texture(size=(*tile.maps.size, 1), format="rgba32float",
                                                usage=U.STORAGE_BINDING | U.TEXTURE_BINDING)
        self.last_point = None  # (x, y) of the last pointer sample
        self.last_stamp = None  # world position of the last stamp
        self.travel = 0.0  # surface distance since the last stamp
        self.prev_hit = None
        self.stamps = 0
        self.unit = None  # the 2D view's grid units (add_texture)
        # Camera frame for the stamp orientation (right vector of the view).
        vp = np.asarray(view.view_proj if view is not None else np.identity(4), np.float64)
        self.right = vp[0, :3] / max(np.linalg.norm(vp[0, :3]), 1e-9)

    def take_dirty(self) -> dict:
        """{UDIM: (x0, y0, x1, y1)} the texels each tile's targets changed since the last call (x1, y1 excluded);
        the tiles untouched are left out."""
        out = {}
        q = self.engine.device.queue
        for i, tile in enumerate(self.tiles):
            w, h = tile.maps.size
            if self._whole[i]:
                out[tile.udim] = (0, 0, w, h)
            else:
                x0, y0, x1, y1 = struct.unpack("4I", q.read_buffer(self.dirty[i]))
                if x0 <= x1:
                    out[tile.udim] = (x0, y0, x1 + 1, y1 + 1)
            self._whole[i] = False
            q.write_buffer(self.dirty[i], 0, EMPTY_RECT)
        return out

    def targets(self) -> list[Target]:
        """Every tile's targets."""
        return [t for tile in self.tiles for t in tile.targets]

    # --- stamps ---------------------------------------------------------------------------------------

    def _radius(self, hit, pressure) -> tuple[float, float]:
        return stamp_radius(self.s, self.view, hit, self.diag, pressure)

    def _hit(self, x: float, y: float):
        r = self.camera.hit(x, y)
        if r is None:
            return None
        p, n = r
        if p[3] == 0 or int(round(p[3])) - 1 != self.material:
            return None
        return p[:3].astype(np.float64), n.astype(np.float64)

    def add(self, points: list[tuple[float, float, float]]) -> int:
        """Pointer samples (x, y, pressure 0-1); returns the stamps placed."""
        stamps = []
        for x, y, pressure in points:
            if self.last_point is None:
                path = [(x, y)]
            else:
                x0, y0 = self.last_point
                n = max(1, int(math.ceil(max(abs(x - x0), abs(y - y0)))))
                path = [(x0 + (x - x0) * k / n, y0 + (y - y0) * k / n) for k in range(1, n + 1)]
            self.last_point = (x, y)
            for px, py in path:
                hit = self._hit(px, py)
                if hit is None:
                    self.prev_hit = None
                    continue
                pos, nrm = hit
                if self.prev_hit is not None:
                    self.travel += float(np.linalg.norm(pos - self.prev_hit))
                self.prev_hit = pos
                rw, rpx = self._radius(pos, pressure)
                step = max(2 * rw * self.s.spacing / 100, 1e-6)
                if self.last_stamp is not None and self.travel < step:
                    continue
                direction = None if self.last_stamp is None else pos - self.last_stamp
                if self.s.follow_path and direction is None:  # the direction needs two stamps
                    self.last_stamp, self.travel = pos, 0.0
                    continue
                stamps += self._stamp(pos, nrm, (px, py), rw, rpx, pressure, direction)
                self.last_stamp, self.travel = pos, 0.0
        if stamps:
            if self.smudge is not None:
                self._smudge(stamps)
            else:
                self._dispatch(stamps)
        return len(stamps)

    def smudge_stamps(self, stamps: list[bytes], mode: int) -> None:
        """Smudge along path: stamps made by the caller (their screen fields hold UVs: the drag is in texture
        space)."""
        self._smudge(stamps, mode)

    def commit(self) -> None:
        """Lay the coverage written by another pass (Filled path) on the targets."""
        self._whole = [True] * len(self.tiles)  # another pass: no rectangle recorded
        for i in range(len(self.tiles)):
            self._lay(i, whole=True)

    def cover_all(self) -> None:
        """Full coverage everywhere (Ribbon path: its per texel values carry the shape in alpha)."""
        self._whole = [True] * len(self.tiles)
        enc = self.engine.device.create_command_encoder()
        for t in self.stroke_texs:
            rp = enc.begin_render_pass(color_attachments=[{"view": t.create_view(), "load_op": "clear",
                                                           "store_op": "store", "clear_value": (1, 0, 0, 1)}])
            rp.end()
        self.engine.device.queue.submit([enc.finish()])

    def _smudge(self, stamps: list[bytes], mode: int | None = None) -> None:
        """Smudge: stamp by stamp, the content where the previous stamp was (the state right before this stamp)
        laid under this one, so the brush drags color along the stroke (within each tile)."""
        d = self.engine.device
        for st in stamps:
            sx, sy = struct.unpack_from("2f", st, 48)
            prev, self._smudge_prev = self._smudge_prev, (sx, sy)
            if prev is None:  # the first stamp has nothing to drag
                continue
            enc = d.create_command_encoder()
            for t in self.stroke_texs:
                rp = enc.begin_render_pass(color_attachments=[{"view": t.create_view(), "load_op": "clear",
                                                               "store_op": "store", "clear_value": (0, 0, 0, 0)}])
                rp.end()
            for t in self.targets():
                t.pages.expand(enc, self.base[t.key])
            d.queue.submit([enc.finish()])
            for tile in self.tiles:
                for t in tile.targets:
                    self.engine.resample(tile.maps, self.view, self.camera, self.base[t.key], self.smudge.get(t.key),
                                         (prev[0] - sx, prev[1] - sy), self.material, self.diag, tile.udim,
                                         dst=t.source)
            self._dispatch([st], mode)

    def _stamp(self, pos, nrm, screen, rw, rpx, pressure, direction) -> list[bytes]:
        s, rnd = self.s, self.rng.random
        n = nrm / max(np.linalg.norm(nrm), 1e-9)
        u = self.right - n * float(np.dot(self.right, n))
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n, [0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        angle = math.radians(s.angle) + (rnd() * 2 - 1) * math.pi * s.angle_jitter
        if s.follow_path and direction is not None:
            d = direction - n * float(np.dot(direction, n))
            if np.linalg.norm(d) > 1e-9:
                v = np.cross(n, u)
                angle += math.atan2(float(np.dot(d, v)), float(np.dot(d, u)))
        k = 1.0 - s.size_jitter * rnd()
        rw, rpx = rw * k, rpx * k
        flow = s.flow / 100 * (pressure if s.flow_pressure else 1.0) * (1.0 - s.flow_jitter * rnd())
        sx, sy = screen
        if s.position_jitter > 0:
            jx, jy = (rnd() * 2 - 1, rnd() * 2 - 1)
            v = np.cross(n, u)
            pos = pos + (u * jx + v * jy) * 2 * rw * s.position_jitter
            sx, sy = sx + jx * 2 * rpx * s.position_jitter, sy + jy * 2 * rpx * s.position_jitter
        out = [struct.pack("16f", *pos, rw, *n, flow, *u, angle, sx, sy, rpx, 0.0)]
        for m, t, reflect, flip in self.symmetry or ():  # the copies of the symmetry (Symmetry pages)
            mp = m @ pos + t
            mn, mu = m @ n, m @ u
            clip = np.asarray(self.view.view_proj, np.float64) @ np.array([*mp, 1.0])
            if clip[3] > 1e-9:
                msx = (clip[0] / clip[3] * 0.5 + 0.5) * self.view.width
                msy = (0.5 - clip[1] / clip[3] * 0.5) * self.view.height
            else:
                msx = msy = -1e9
            mrw, mrpx = stamp_radius(self.s, self.view, mp, self.diag, pressure)
            mrw, mrpx = mrw * k, mrpx * k
            # A reflection flips the stamp's frame: the same angle seen in it is the mirror image. Flip copy (U or V)
            # mirrors the alpha in the stamp's own frame (encoded in the angle, brush.wgsl).
            a = -angle if reflect else angle
            out.append(struct.pack("16f", *mp, mrw, *mn, flow, *mu, a + FLIP_CODE * flip, msx, msy, mrpx, 0.0))
        return out

    def add_texture(self, points: list[tuple[float, float, float]], radius: float, unit) -> int:
        """Painting in the 2D view: pointer samples in units of the UV grid (x, y, pressure; `unit` (w, h) units
        the 0-1 square, the tile U, V at x + U * w, y - V * h: the Texture Set's size), stamps of `radius` units
        every Spacing along the path in texture space (no mesh, no culling); each tile takes them at its own
        resolution."""
        self.unit = tuple(unit)
        s, rnd = self.s, self.rng.random
        stamps = []
        for x, y, pressure in points:
            if self.last_point is None:
                path = [(x, y)]
            else:
                x0, y0 = self.last_point
                n = max(1, int(math.ceil(max(abs(x - x0), abs(y - y0)))))
                path = [(x0 + (x - x0) * k / n, y0 + (y - y0) * k / n) for k in range(1, n + 1)]
            self.last_point = (x, y)
            for px, py in path:
                if self.prev_hit is not None:
                    self.travel += float(math.hypot(px - self.prev_hit[0], py - self.prev_hit[1]))
                self.prev_hit = (px, py)
                r = radius * (pressure if s.size_pressure else 1.0)
                step = max(2 * r * s.spacing / 100, 0.5)
                if self.last_stamp is not None and self.travel < step:
                    continue
                angle = math.radians(s.angle) + (rnd() * 2 - 1) * math.pi * s.angle_jitter
                k = 1.0 - s.size_jitter * rnd()
                flow = s.flow / 100 * (pressure if s.flow_pressure else 1.0) * (1.0 - s.flow_jitter * rnd())
                jx, jy = (rnd() * 2 - 1, rnd() * 2 - 1) if s.position_jitter > 0 else (0.0, 0.0)
                cx, cy = px + jx * 2 * r * s.position_jitter, py + jy * 2 * r * s.position_jitter
                stamps.append(struct.pack("16f", 0, 0, 0, 1, 0, 0, 1, flow, 1, 0, 0, angle, cx, cy, r * k, 0.0))
                self.last_stamp, self.travel = (px, py), 0.0
        if stamps:
            self._dispatch(stamps, mode=4)
        return len(stamps)

    def fill(self, uv_triangles: np.ndarray) -> int:
        """Polygon Fill: triangles ((N, 3, 2) UVs) fully covered, then laid on the targets like stamps."""
        if not len(uv_triangles):
            return 0
        d = self.engine.device
        enc = d.create_command_encoder()
        for i, tile in enumerate(self.tiles):  # each tile takes the triangles in its square (the rest is clipped)
            vb = d.create_buffer_with_data(
                data=np.ascontiguousarray(local_uv(uv_triangles, tile.udim), np.float32).tobytes(),
                usage=wgpu.BufferUsage.VERTEX)
            rp = enc.begin_render_pass(color_attachments=[{"view": self.stroke_texs[i].create_view(),
                                                           "load_op": "load", "store_op": "store"}])
            rp.set_pipeline(self.engine.fill_pipe)
            rp.set_vertex_buffer(0, vb)
            rp.draw(len(uv_triangles) * 3)
            rp.end()
            self._whole[i] = True  # a render pass: no rectangle recorded
        d.queue.submit([enc.finish()])
        for i in range(len(self.tiles)):
            self._lay(i, whole=True)
        return len(uv_triangles)

    def _reaching(self, stamps: list[bytes], tile: TileTargets, mode) -> list[bytes]:
        """The stamps that reach a tile: on the surface those whose sphere meets the tile's triangles' bounds; in
        texture space (mode 4) those over its square, moved into its texels."""
        w, h = tile.maps.size
        out = []
        if mode == 4:  # from the grid's units to this tile's texels
            du, dv = tile_uv(tile.udim)
            kx, ky = w / self.unit[0], h / self.unit[1]
            for st in stamps:
                v = list(struct.unpack("16f", st))
                v[12], v[13], v[14] = (v[12] - du * self.unit[0]) * kx, (v[13] + dv * self.unit[1]) * ky, v[14] * kx
                r = v[14]
                if -r <= v[12] <= w + r and -r <= v[13] <= h + r:
                    out.append(struct.pack("16f", *v))
            return out
        if tile.maps.bounds is None:
            return out
        lo, hi = tile.maps.bounds
        for st in stamps:
            x, y, z, r = struct.unpack_from("4f", st, 0)
            if np.all(np.asarray((x, y, z)) >= lo - r) and np.all(np.asarray((x, y, z)) <= hi + r):
                out.append(st)
        return out

    def _dispatch(self, stamps: list[bytes], mode: int | None = None) -> None:
        d = self.engine.device
        if mode is None:
            mode = ALIGNMENTS.index(self.s.alignment)
        cull = math.cos(math.radians(min(max(self.s.backface_culling, 0.0), 180.0)))
        tol = self.diag * 2e-3
        v = self.view
        vp = np.identity(4, np.float32) if v is None else v.view_proj
        eye, vw, vh = ((0.0, 0.0, 0.0), 1, 1) if v is None else (v.eye, v.width, v.height)
        pipe = self.engine.brush_pipe
        for i, tile in enumerate(self.tiles):
            mine = self._reaching(stamps, tile, mode)
            w, h = tile.maps.size
            cam_tex = self.camera.texture if self.camera is not None else tile.maps.position
            for k in range(0, len(mine), MAX_STAMPS):
                chunk = mine[k:k + MAX_STAMPS]
                uni = np.ascontiguousarray(np.asarray(vp, np.float32).T).tobytes() + struct.pack(
                    "12f", *eye, 1.0 if self.s.stamps_blending == "Lighten" else 0.0, vw, vh, tol, self.blocks[i][0], mode, len(chunk), cull, 0.5)
                ub = d.create_buffer_with_data(data=uni, usage=wgpu.BufferUsage.UNIFORM)
                sb = d.create_buffer_with_data(data=b"".join(chunk), usage=wgpu.BufferUsage.STORAGE)
                group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
                    {"binding": 0, "resource": {"buffer": ub}}, {"binding": 1, "resource": {"buffer": sb}},
                    {"binding": 2, "resource": tile.maps.position.create_view()},
                    {"binding": 3, "resource": tile.maps.normal.create_view()},
                    {"binding": 4, "resource": cam_tex.create_view()},
                    {"binding": 5, "resource": self.alpha_tex.create_view()},
                    {"binding": 6, "resource": self.engine.sampler},
                    {"binding": 7, "resource": self.stroke_texs[i].create_view()},
                    {"binding": 8, "resource": {"buffer": self.dirty[i]}},
                    {"binding": 9, "resource": {"buffer": self.blocks[i][2]}}])
                enc = d.create_command_encoder()
                cp = enc.begin_compute_pass()
                cp.set_pipeline(pipe)
                cp.set_bind_group(0, group)
                cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
                cp.end()
                d.queue.submit([enc.finish()])
                self._lay(i)
        self.stamps += len(stamps)

    def _lay(self, i: int, whole: bool = False) -> None:
        """Lay tile `i`'s coverage on its targets where this pass changed it: the pages the brush marked (every page
        after a pass that writes the coverage another way: Polygon Fill, paths), each saved first the first time
        the stroke changes it, added where nothing was painted (not by the Eraser: nothing to erase there). A
        Smudge lays them over the state before the stamp."""
        d = self.engine.device
        tile = self.tiles[i]
        size, per_row, bits = self.blocks[i]
        w, h = tile.maps.size
        count = per_row * -(-h // size)
        if whole:
            changed = list(range(count))
        else:
            words = np.frombuffer(d.queue.read_buffer(bits), np.uint8)
            d.queue.write_buffer(bits, 0, bytes(len(words)))
            changed = [int(k) for k in np.nonzero(np.unpackbits(words, bitorder="little"))[0] if k < count]
        if not changed:
            return
        added = {t.key: (set() if t.erase else set(t.pages.allocate(changed))) for t in tile.targets}
        enc = d.create_command_encoder()
        for t in tile.targets:
            self.saved.save(enc, t, changed, added[t.key])
        xs, ys = [k % per_row for k in changed], [k // per_row for k in changed]
        area = (min(xs) * size, min(ys) * size, min((max(xs) + 1) * size, w), min((max(ys) + 1) * size, h))
        for t in tile.targets:
            self._commit(enc, t, i, area)
        d.queue.submit([enc.finish()])

    def _commit(self, enc, t: Target, i: int, area) -> None:
        """One commit.wgsl pass on target `t` of tile `i` over `area` (x0, y0, x1, y1): the pages the stroke changed,
        over what they held before it (or, Smudge, over the state before the stamp)."""
        d = self.engine.device
        tile = self.tiles[i]
        pipe = self.engine.commit_pipe(t.pages.format)
        value = tuple(t.value) + (1.0,) * (4 - len(t.value))
        size, per_row, _bits = self.blocks[i]
        x0, y0, x1, y1 = area
        base = self.base[t.key] if self.base is not None else None
        opacity, mode, flags = self.s.stroke_opacity / 100, -1.0, 0.0
        if self.blending is not None and not t.erase:  # blend_lib's lay with the channel's own rules
            from trafilatura3d.paint import blending
            mode, opacity = float(blending.INDEX[self.blending[0]]), self.blending[1]
            flags = 1.0 * t.color + 2.0 * (not t.hdr) + 4.0 * (t.key[1] == "normal")
        ub = d.create_buffer_with_data(data=struct.pack("8f4I4f", *value[:3], 1.0 if tile.stencil is not None else 0.0,
                                                        opacity, 1.0 if t.erase else 0.0,
                                                        1.0 if t.source is not None else 0.0,
                                                        1.0 if tile.limit is not None else 0.0,
                                                        x0, y0, size, per_row, 1.0 if base is not None else 0.0,
                                                        mode, flags, 0.0),
                                       usage=wgpu.BufferUsage.UNIFORM)
        saved, table = self.saved.of(t.key, t.pages)
        group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": ub}},
            {"binding": 1, "resource": (base or self.engine.no_source).create_view()},
            {"binding": 2, "resource": self.stroke_texs[i].create_view()},
            {"binding": 3, "resource": t.pages.array.create_view(dimension="2d-array")},
            {"binding": 4, "resource": (t.source or self.engine.no_source).create_view()},
            {"binding": 5, "resource": (tile.limit or self.engine.no_source).create_view()},
            {"binding": 6, "resource": (tile.stencil or self.engine.no_source).create_view()},
            {"binding": 7, "resource": saved.create_view(dimension="2d-array")},
            {"binding": 8, "resource": {"buffer": table}}])
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, group)
        cp.dispatch_workgroups((x1 - x0 + 7) // 8, (y1 - y0 + 7) // 8)
        cp.end()

    def end(self) -> Saved:
        """What the entries held before the stroke, for the undo (the caller destroys it)."""
        for t in self.stroke_texs:
            t.destroy()
        for b in self.dirty:
            b.destroy()
        for _size, _per_row, bits in self.blocks:
            bits.destroy()
        for tex in (self.base or {}).values():
            tex.destroy()
        return self.saved
