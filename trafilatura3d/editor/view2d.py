"""The 2D View: a node's output drawn by wgpu straight into the widget (no readback),
a top toolbar (Save Image, Copy Image, Select output, Image Information, Histogram) and a display
toolbar (color channels, checkerboard, tiled display, fit, 1:1, zoom), with the resolution and
format at the bottom left. Keys (with the view focused): C R G B A channels, Space tiling, F fit,
Z 1:1; wheel zooms at the cursor, any drag pans."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import wgpu
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QToolBar,
                               QVBoxLayout, QWidget)

from trafilatura3d import imageio
from trafilatura3d.editor import theme
from trafilatura3d.editor.canvas import Canvas
from trafilatura3d.editor.icons import icon
from trafilatura3d.editor.render3d import MAX_FPS, present_method
from trafilatura3d.evaluator import BYTES_PER_PIXEL, NUMPY
from trafilatura3d.export import convert

SHADER = """
struct View { center: vec2f, zoom: f32, gray: u32, size: vec2f, channel: u32, flags: u32 };
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var lut: texture_3d<f32>;  // color management: image space -> display / view
@group(0) @binding(3) var lut_samp: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn checker(frag: vec2f, light: f32) -> vec3f {
    let k = (i32(floor(frag.x / 10.0)) + i32(floor(frag.y / 10.0))) % 2;
    return vec3f(select(light * 0.75, light, k == 0));
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
    let d = vec2f(textureDimensions(tex));
    var p = (frag.xy - view.size * 0.5) / view.zoom + view.center;
    let tiled = (view.flags & 2u) != 0u;
    if (!tiled && (p.x < 0.0 || p.y < 0.0 || p.x >= d.x || p.y >= d.y)) {
        return vec4f(0.137, 0.137, 0.137, 1.0);
    }
    p = ((p % d) + d) % d;
    var c = textureLoad(tex, vec2i(p), 0);
    if (view.gray == 1u) { c = vec4f(c.rrr, 1.0); }
    let n = f32(textureDimensions(lut).x);
    let managed = textureSampleLevel(lut, lut_samp, clamp(c.rgb, vec3f(0.0), vec3f(1.0)) * ((n - 1.0) / n) + 0.5 / n, 0.0).rgb;
    if ((view.flags & 4u) != 0u) { c = vec4f(managed, c.a); }
    switch view.channel {
        case 1u: { return vec4f(c.rrr, 1.0); }
        case 2u: { return vec4f(c.ggg, 1.0); }
        case 3u: { return vec4f(c.bbb, 1.0); }
        case 4u: { return vec4f(c.aaa, 1.0); }
        default: {}
    }
    if ((view.flags & 1u) != 0u) {  // transparency over a checkerboard
        return vec4f(mix(checker(frag.xy, 0.45), clamp(c.rgb, vec3f(0.0), vec3f(1.0)), clamp(c.a, 0.0, 1.0)), 1.0);
    }
    return vec4f(c.rgb, 1.0);
}
"""

# UV wireframe of a 3D View mesh over the image ("Display UV's in 2D View").
UV_SHADER = """
struct V { center: vec2f, zoom: f32, pad: f32, size: vec2f, tex: vec2f, color: vec4f };
@group(0) @binding(0) var<uniform> v: V;
@vertex
fn vs(@location(0) uv: vec2f) -> @builtin(position) vec4f {
    let frag = (uv * v.tex - v.center) * v.zoom + v.size * 0.5;
    return vec4f(frag.x / v.size.x * 2.0 - 1.0, 1.0 - frag.y / v.size.y * 2.0, 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4f { return v.color; }
"""

# Overlay lines in the widget's physical pixels, each vertex with its color (cursors, gizmos).
OVERLAY_SHADER = """
@group(0) @binding(0) var<uniform> size: vec4f;
struct Out { @builtin(position) pos: vec4f, @location(0) color: vec4f };
@vertex
fn vs(@location(0) px: vec2f, @location(1) color: vec4f) -> Out {
    return Out(vec4f(px.x / size.x * 2.0 - 1.0, 1.0 - px.y / size.y * 2.0, 0.0, 1.0), color);
}
@fragment
fn fs(i: Out) -> @location(0) vec4f { return i.color; }
"""
UV_EDGES_MAX = 400_000  # dense primitives: every few edges is enough to read the layout

CHANNELS = ["RGB", "Red", "Green", "Blue", "Alpha"]


class Viewport(Canvas):
    hover = Signal(float, float)  # texel under the cursor (texture pixels)

    def __init__(self, device, parent=None):
        super().__init__(parent, present_method=present_method(), update_mode="ondemand", max_fps=MAX_FPS)
        self.device = device
        self.source = None  # callable -> (texture, is_gray) or None
        self.zoom = None
        self.center = (0.0, 0.0)
        self.channel = 0
        self.checker = True
        self.tiled = False
        self._fitted = None
        self._drag = None
        ctx = self.get_wgpu_context()
        self.format = ctx.get_preferred_format(device.adapter).replace("-srgb", "")  # raw values, as written to files
        ctx.configure(device=device, format=self.format)
        module = device.create_shader_module(code=SHADER)
        self.pipeline = device.create_render_pipeline(
            layout="auto", vertex={"module": module, "entry_point": "vs"},
            fragment={"module": module, "entry_point": "fs", "targets": [{"format": self.format}]},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list})
        uv_module = device.create_shader_module(code=UV_SHADER)
        blend = {"color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        self.uv_pipeline = device.create_render_pipeline(
            layout="auto", vertex={"module": uv_module, "entry_point": "vs", "buffers": [
                {"array_stride": 8, "attributes": [{"shader_location": 0, "offset": 0, "format": "float32x2"}]}]},
            fragment={"module": uv_module, "entry_point": "fs", "targets": [{"format": self.format, "blend": blend}]},
            primitive={"topology": wgpu.PrimitiveTopology.line_list})
        self.uv_uniform = device.create_buffer(size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.uv_lines = None  # (buffer, vertex count)
        self.uv_color = (1.0, 0.62, 0.15, 0.85)
        ov_module = device.create_shader_module(code=OVERLAY_SHADER)
        self.overlay_pipeline = device.create_render_pipeline(
            layout="auto", vertex={"module": ov_module, "entry_point": "vs", "buffers": [
                {"array_stride": 24, "attributes": [{"shader_location": 0, "offset": 0, "format": "float32x2"},
                                                    {"shader_location": 1, "offset": 8, "format": "float32x4"}]}]},
            fragment={"module": ov_module, "entry_point": "fs", "targets": [{"format": self.format, "blend": blend}]},
            primitive={"topology": wgpu.PrimitiveTopology.line_list})
        self.overlay_uniform = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.lut = self._lut_texture(np.ones((1, 1, 1, 4), np.float32))  # OCIO display LUT (identity: off)
        self.managed = False
        self.lut_sampler = device.create_sampler(mag_filter="linear", min_filter="linear")
        self.show_uvs = True
        self.add_event_handler(self._on_event, "wheel", "pointer_down", "pointer_move", "pointer_up", "double_click")
        self.request_draw(self._draw)

    def refresh(self):
        self.request_draw(self._draw)

    def set_uv_lines(self, segments) -> None:
        """UV segments ((N, 2, 2), UV 0-1 over the image) drawn over it, or None."""
        if segments is None or not len(segments):
            self.uv_lines = None
        else:
            if len(segments) > UV_EDGES_MAX:
                segments = segments[:: len(segments) // UV_EDGES_MAX + 1]
            lines = np.ascontiguousarray(np.asarray(segments).reshape(-1, 2), np.float32)
            buf = self.device.create_buffer_with_data(data=lines.tobytes(), usage=wgpu.BufferUsage.VERTEX)
            self.uv_lines = (buf, len(lines))
        self.refresh()

    def overlay(self) -> np.ndarray | None:
        """Lines drawn last, in physical pixels: (N, 6) vertices x, y, r, g, b, a in pairs; none here."""
        return None

    def _lut_texture(self, lut):
        n = lut.shape[0]
        t = self.device.create_texture(size=(n, n, n), dimension="3d", format="rgba16float",
                                       usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": t}, np.ascontiguousarray(lut, np.float16).tobytes(),
                                        {"bytes_per_row": n * 8, "rows_per_image": n}, (n, n, n))
        return t

    def set_lut(self, lut):
        """A display LUT ((N, N, N, 4) float) or None for Legacy (values shown as they are)."""
        self.managed = lut is not None
        if lut is not None:
            self.lut = self._lut_texture(lut)
        self.refresh()

    def texture_size(self):
        shown = self.source() if self.source else None
        return (shown[0].width, shown[0].height) if shown else None

    def shown(self) -> list:
        """What the view draws: [(texture, is_gray, (x, y) offset, texels per unit)], offsets in `unit` texels;
        here the source alone (the Paint 2D view lays UDIM tiles side by side, each at its resolution)."""
        got = self.source() if self.source else None
        return [(got[0], got[1], (0, 0), 1.0)] if got else []

    def unit(self) -> tuple | None:
        """The size of the 0-1 UV square in view units (the UV wireframe's scale): here the shown texture's."""
        return self.texture_size()

    def extent(self) -> tuple:
        """The rectangle (x0, y0, x1, y1) Fit frames, in view units."""
        w, h = self.unit()
        return 0, 0, w, h

    def fit(self):
        self.zoom = None
        self.refresh()

    def one_to_one(self):
        self.zoom = 1.0 * self.get_pixel_ratio()
        self.refresh()

    def _draw(self):
        ctx = self.get_wgpu_context()
        target = ctx.get_current_texture().create_view()
        shown = self.shown()
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(color_attachments=[{"view": target, "load_op": wgpu.LoadOp.clear,
                                                       "store_op": wgpu.StoreOp.store,
                                                       "clear_value": (0.137, 0.137, 0.137, 1.0)}])
        if shown:
            w, h = self.get_physical_size()
            uw, uh = self.unit()
            ext = self.extent()
            if self.zoom is None or self._fitted != ext:
                self.zoom = min(w / (ext[2] - ext[0]), h / (ext[3] - ext[1])) * 0.9
                self.center = ((ext[0] + ext[2]) / 2, (ext[1] + ext[3]) / 2)
                self._fitted = ext
            flags = int(self.checker) | (int(self.tiled) << 1) | (int(self.managed) << 2)
            rp.set_pipeline(self.pipeline)
            for t, gray, (ox, oy), k in shown:  # each texture drawn in its own rectangle, at its scale
                x0, y0 = (ox - self.center[0]) * self.zoom + w / 2, (oy - self.center[1]) * self.zoom + h / 2
                x1, y1 = x0 + t.width / k * self.zoom, y0 + t.height / k * self.zoom
                sx0, sy0 = max(int(x0), 0), max(int(y0), 0)
                sx1, sy1 = min(int(math.ceil(x1)), w), min(int(math.ceil(y1)), h)
                if self.tiled and len(shown) == 1:
                    sx0, sy0, sx1, sy1 = 0, 0, w, h
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                ub = self.device.create_buffer_with_data(data=struct.pack(
                    "2f f I 2f I I", (self.center[0] - ox) * k, (self.center[1] - oy) * k, self.zoom / k, int(gray), w,
                    h, self.channel, flags), usage=wgpu.BufferUsage.UNIFORM)
                group = self.device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=[
                    {"binding": 0, "resource": {"buffer": ub}}, {"binding": 1, "resource": t.create_view()},
                    {"binding": 2, "resource": self.lut.create_view()}, {"binding": 3, "resource": self.lut_sampler}])
                rp.set_bind_group(0, group)
                rp.set_scissor_rect(sx0, sy0, sx1 - sx0, sy1 - sy0)
                rp.draw(3)
            rp.set_scissor_rect(0, 0, w, h)
            if self.uv_lines is not None and self.show_uvs:
                self.device.queue.write_buffer(self.uv_uniform, 0, struct.pack(
                    "2f f f 2f 2f 4f", *self.center, self.zoom, 0.0, w, h, uw, uh, *self.uv_color))
                ug = self.device.create_bind_group(layout=self.uv_pipeline.get_bind_group_layout(0), entries=[
                    {"binding": 0, "resource": {"buffer": self.uv_uniform}}])
                rp.set_pipeline(self.uv_pipeline)
                rp.set_bind_group(0, ug)
                rp.set_vertex_buffer(0, self.uv_lines[0])
                rp.draw(self.uv_lines[1])
        lines = self.overlay()
        if lines is not None and len(lines):
            w, h = self.get_physical_size()
            self.device.queue.write_buffer(self.overlay_uniform, 0, struct.pack("4f", w, h, 0.0, 0.0))
            og = self.device.create_bind_group(layout=self.overlay_pipeline.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": {"buffer": self.overlay_uniform}}])
            buf = self.device.create_buffer_with_data(data=np.ascontiguousarray(lines, np.float32).tobytes(),
                                                      usage=wgpu.BufferUsage.VERTEX)
            rp.set_pipeline(self.overlay_pipeline)
            rp.set_bind_group(0, og)
            rp.set_vertex_buffer(0, buf)
            rp.draw(len(lines))
        rp.end()
        self.device.queue.submit([enc.finish()])

    def _on_event(self, e):
        ratio = self.get_pixel_ratio()
        x, y = e["x"] * ratio, e["y"] * ratio
        kind = e["event_type"]
        if self.zoom is not None:
            w, h = self.get_physical_size()
            self.hover.emit((x - w / 2) / self.zoom + self.center[0], (y - h / 2) / self.zoom + self.center[1])
        if self.zoom is None:
            return
        if kind == "double_click":
            self.zoom = None
        elif kind == "wheel":
            w, h = self.get_physical_size()
            px = (x - w / 2) / self.zoom + self.center[0]
            py = (y - h / 2) / self.zoom + self.center[1]
            from trafilatura3d.editor.preferences import zoom_sign
            self.zoom *= 1.0015 ** (-e["dy"] * zoom_sign())
            self.center = (px - (x - w / 2) / self.zoom, py - (y - h / 2) / self.zoom)
        elif kind == "pointer_down":
            self._drag = (x, y, self.center)
            return
        elif kind == "pointer_move" and self._drag:
            x0, y0, (cx, cy) = self._drag
            self.center = (cx - (x - x0) / self.zoom, cy - (y - y0) / self.zoom)
        elif kind == "pointer_up":
            self._drag = None
            return
        else:
            return
        self.refresh()


class View2D(QWidget):
    title_changed = Signal(str)
    zoom_changed = Signal()

    def __init__(self, document):
        super().__init__()
        self.doc = document
        self.package, self.graph_id, self.node_id = None, "", None
        self._last = None  # (job, output) of the texture drawn last (`_texture`)
        self._asked = None  # the key of the result the refresh was asked for, not cooked yet (`_texture`)
        self.output = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        top = theme.divider(QToolBar(), "bottom")
        top.setIconSize(QSize(16, 16))
        top.addAction(icon("save"), "Save Image…", self.save_image)
        top.addAction(icon("copy"), "Copy Image to Clipboard", self.copy_image)
        self.outputs = QComboBox()
        self.outputs.setToolTip("Select output")
        self.outputs.currentTextChanged.connect(self._output_changed)
        top.addWidget(self.outputs)
        self.act_info = top.addAction(icon("info"), "Image Information")
        self.act_info.setCheckable(True)
        self.act_info.toggled.connect(self._toggle_info)
        self.act_hist = top.addAction(icon("histogram"), "Display Histogram")
        self.act_hist.setCheckable(True)
        self.act_hist.toggled.connect(self._toggle_hist)
        self.act_uvs = top.addAction(icon("chain"), "Display UV's")
        self.act_uvs.setCheckable(True)
        self.act_uvs.toggled.connect(self._toggle_uvs)
        self.act_uvs.setVisible(False)
        lay.addWidget(top)
        self.view = Viewport(self.doc.ev.device)
        self.view.source = self._texture
        self.view.hover.connect(self._hover)
        lay.addWidget(self.view, 1)
        self.info_label = QLabel()
        self.info_label.setVisible(False)
        self.hist_label = QLabel()
        self.hist_label.setVisible(False)
        self.hist_label.setFixedHeight(80)
        lay.addWidget(self.info_label)
        lay.addWidget(self.hist_label)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(6, 2, 6, 2)
        self.format_label = QLabel("")
        bottom.addWidget(self.format_label)
        bottom.addStretch(1)
        bar = QToolBar()
        bar.setIconSize(QSize(16, 16))
        self.channel_box = QComboBox()
        self.channel_box.addItems(CHANNELS)
        self.channel_box.setToolTip("Color channels")
        self.channel_box.currentIndexChanged.connect(self._channel)
        bar.addWidget(self.channel_box)
        self.cs_box = QComboBox()
        self.cs_box.setToolTip("Color space (display / view)")
        self.cs_box.currentIndexChanged.connect(lambda _i: self._apply_color())
        bar.addWidget(self.cs_box)
        chk = bar.addAction(icon("checker"), "Show checkerboard")
        chk.setCheckable(True)
        chk.toggled.connect(self._checker)
        chk.setChecked(True)
        self.act_tile = bar.addAction(icon("tile"), "Tiled display (Space)")
        self.act_tile.setCheckable(True)
        self.act_tile.toggled.connect(self._tile)
        bar.addAction(icon("fit"), "Fit to view (F)", lambda: (self.view.fit(), self._zoom_text()))
        bar.addAction(icon("reset_zoom"), "Fit to scale 1:1 (Z)", lambda: (self.view.one_to_one(), self._zoom_text()))
        self.zoom_field = QLineEdit()
        self.zoom_field.setFixedWidth(64)
        self.zoom_field.editingFinished.connect(self._zoom_typed)
        bar.addWidget(self.zoom_field)
        bottom.addWidget(bar)
        holder = theme.divider(QWidget(), "top")
        holder.setLayout(bottom)
        lay.addWidget(holder)
        for key, idx in (("C", 0), ("R", 1), ("G", 2), ("B", 3), ("A", 4)):
            QShortcut(QKeySequence(key), self, activated=lambda idx=idx: self.channel_box.setCurrentIndex(idx),
                      context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Space"), self, activated=self.act_tile.trigger, context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("F"), self, activated=lambda: (self.view.fit(), self._zoom_text()),
                  context=Qt.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Z"), self, activated=lambda: (self.view.one_to_one(), self._zoom_text()),
                  context=Qt.WidgetWithChildrenShortcut)
        self._zoom_timer = QTimer(self, interval=300, timeout=self._zoom_text)
        self._zoom_timer.start()

    # --- UVs of a 3D View mesh -------------------------------------------------------------------

    def set_uvs(self, data):
        """(uv (n, 2), triangles (m, 3)) of a 3D View mesh drawn over the image, or None."""
        if data is None:
            self.view.uv_lines = None
            self.act_uvs.setVisible(False)
        else:
            uv, tri = data
            edges = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1), axis=0)
            self.view.set_uv_lines(uv[edges])
            self.act_uvs.setVisible(True)
            self.act_uvs.setChecked(True)
            self.view.show_uvs = True
        self.view.refresh()

    def _toggle_uvs(self, on):
        self.view.show_uvs = on
        self.view.refresh()

    # --- target -------------------------------------------------------------------------------

    def show_node(self, package, graph_id, node_id):
        self.package, self.graph_id, self.node_id = package, graph_id, node_id
        self.outputs.blockSignals(True)
        self.outputs.clear()
        job_outputs = self._outputs()
        self.outputs.addItems(list(job_outputs))
        self.outputs.setVisible(len(job_outputs) > 1)
        self.output = next(iter(job_outputs), None)
        self.outputs.blockSignals(False)
        self.refresh()

    def _outputs(self) -> dict:
        if self.package is None or self.node_id is None:
            return {}
        compiled = self.doc.shown(self.package, self.graph_id)
        if compiled is None or self.node_id not in compiled.nodes:
            return {}
        return compiled.nodes[self.node_id]

    def _output_changed(self, name):
        self.output = name or None
        self.refresh()

    def source(self):
        """(job, output) of what the view shows, or (None, None): what a refresh cooks (`LiveCook` steps; the
        Graph workspace's refresh calls `refresh` after its edits)."""
        outs = self._outputs()
        if not outs:
            return None, None
        key = self.output if self.output in outs else next(iter(outs))
        return outs[key]

    def _texture(self):
        """The texture drawn: the node's result when cooked; else the refresh (`LiveCook`, its 2D step) computes it
        in steps and redraws the view, the result shown before staying meanwhile (cooking it here, in the draw,
        stalled the interface for the whole chain). The one drawn is held: the cache keeps it while on view."""
        job, out = self.source()
        if job is None:
            return None
        ev = self.doc.ev
        if job.key not in ev.cache and self.doc.live is not None:
            if self._asked != job.key:  # once: every draw asking again restarted the refresh before it was done
                self._asked = job.key
                self.doc.live.schedule()
            job, out = self._last if self._last is not None and self._last[0].key in ev.cache else (None, None)
            if job is None:
                return None
        else:
            self._asked = None
        try:
            texture = ev.result(job, out)
        except Exception as e:
            self.doc.error.emit(f"Cook error: {e}")
            return None
        self._last = (job, out)
        ev.hold("view 2d", [job])
        return texture, job.is_gray(out)

    def fill_color_spaces(self):
        """Legacy: values as they are; OCIO: every display / view of the config."""
        from trafilatura3d import colormanagement as cm
        self.cs_box.blockSignals(True)
        self.cs_box.clear()
        if cm.active():
            d0, v0 = cm.default_display_view()
            for d in cm.displays():
                for v in cm.views(d):
                    self.cs_box.addItem(f"{d} / {v}", (d, v))
            self.cs_box.setCurrentIndex(max(0, self.cs_box.findText(f"{d0} / {v0}")))
        else:
            self.cs_box.addItem("sRGB (default)", None)
        self.cs_box.blockSignals(False)
        self._apply_color()

    def source_space(self, job, out):
        """The color space of the shown image: its Output node's Color space attribute, else an sRGB
        texture for color and raw data for grayscale (project settings)."""
        from trafilatura3d import colormanagement as cm
        node = self.package.graphs[self.graph_id].nodes.get(self.node_id) if self.package else None
        if node is not None and node.type == "output":
            cs = self.package.definition(node).values(node).get("color_space", "")
            if cm.known(cs):
                return cs
        return cm.data_space() if job.is_gray(out) else cm.texture_space()

    def _apply_color(self):
        from trafilatura3d import colormanagement as cm
        dv = self.cs_box.currentData()
        job, out = self.source()
        if not cm.active() or dv is None or job is None:
            self.view.set_lut(None)
            return
        try:
            self.view.set_lut(cm.display_lut(self.source_space(job, out), *dv))
        except Exception as e:  # noqa: BLE001 - a bad config shows the raw values and says why
            self.doc.error.emit(f"Color management: {e}")
            self.view.set_lut(None)

    def refresh(self):
        job, out = self.source()
        if self.cs_box.count() == 0:
            self.fill_color_spaces()
        else:
            self._apply_color()
        if job is None:
            self.format_label.setText("")
            self.title_changed.emit("2D View")
        else:
            depth = {"8": "8bpc", "16": "16bpc", "16f": "16F", "32f": "32F"}[job.depths[out]]
            self.format_label.setText(f"{job.size[0]} x {job.size[1]} ({'L' if job.is_gray(out) else 'RGBA'}, {depth})")
            node = self.package.graphs[self.graph_id].nodes.get(self.node_id)
            label = self.node_id
            if node is not None:
                d = self.package.definition(node)
                v = d.values(node)
                label = (v.get("label") or v.get("identifier")) if node.type == "output" else d.label
            self.title_changed.emit(f"{label} - 2D View")
        self.view.refresh()
        if self.act_hist.isChecked():
            self._histogram()

    # --- toolbar ------------------------------------------------------------------------------

    def _channel(self, idx):
        self.view.channel = idx
        self.view.refresh()

    def _checker(self, on):
        self.view.checker = on
        self.view.refresh()

    def _tile(self, on):
        self.view.tiled = on
        self.view.refresh()

    def _zoom_text(self):
        if self.view.zoom and not self.zoom_field.hasFocus():
            self.zoom_field.setText(f"{self.view.zoom / self.view.get_pixel_ratio() * 100:.2f}%")

    def _zoom_typed(self):
        try:
            z = float(self.zoom_field.text().strip().rstrip("%")) / 100
        except ValueError:
            return
        if z > 0:
            self.view.zoom = z * self.view.get_pixel_ratio()
            self.view.refresh()

    def _read(self):
        job, out = self.source()
        if job is None:
            return None, None
        return job, self.doc.ev.read(job, out)

    def save_image(self):
        job, pixels = self._read()
        if job is None:
            return
        name, _ = QFileDialog.getSaveFileName(self, "Save Image", f"{self.node_id}.png",
                                              "PNG (*.png);;TGA (*.tga);;EXR (*.exr);;TIFF (*.tif)")
        if not name:
            return
        ext = name.rsplit(".", 1)[-1].lower()
        _, out = self.source()
        imageio.write(Path(name), convert(pixels, job.depths[out], ext))

    def copy_image(self):
        job, out = self.source()
        if job is None:
            return
        img = self.doc.ev.thumbnails([(job, out)], max(job.size))[0]
        q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 4, QImage.Format_RGBA8888).copy()
        QApplication.clipboard().setImage(q)

    def _toggle_info(self, on):
        self.info_label.setVisible(on)

    def _hover(self, x, y):
        if not self.act_info.isChecked():
            return
        job, out = self.source()
        if job is None:
            return
        w, h = job.size
        if not (0 <= x < w and 0 <= y < h):
            self.info_label.setText("")
            return
        t = self.doc.ev.result(job, out)
        bpp = BYTES_PER_PIXEL[t.format]
        data = self.doc.ev.device.queue.read_texture({"texture": t, "origin": (int(x), int(y), 0)},
                                                     {"bytes_per_row": bpp}, (1, 1, 1))
        dtype, n = NUMPY[t.format]
        v = np.frombuffer(data, dtype)[:n].astype(np.float64)
        if np.dtype(dtype).kind == "u":  # normalized integers (8 or 16 bit)
            v = v / float(np.iinfo(dtype).max)
        vals = "  ".join(f"{c}: {x:.4f}" for c, x in zip("RGBA" if n == 4 else "L", v))
        self.info_label.setText(f"  X: {int(x)}  Y: {int(y)}    {vals}")

    def _toggle_hist(self, on):
        self.hist_label.setVisible(on)
        if on:
            self._histogram()

    def _histogram(self):
        job, out = self.source()
        if job is None:
            return
        img = self.doc.ev.thumbnails([(job, out)], 256)[0].astype(np.float64)
        lum = img[..., 0] if job.is_gray(out) else img[..., :3] @ np.array([0.299, 0.587, 0.114])
        hist, _ = np.histogram(lum, bins=128, range=(0, 255))
        w, h = 256, 80
        canvas = np.full((h, w, 4), 35, np.uint8)
        canvas[..., 3] = 255
        top = hist.max() or 1
        for i, c in enumerate(hist):
            bar = int(c / top * (h - 4))
            canvas[h - bar:, i * 2:i * 2 + 2, :3] = 200
        q = QImage(canvas.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        from PySide6.QtGui import QPixmap
        self.hist_label.setPixmap(QPixmap.fromImage(q))
