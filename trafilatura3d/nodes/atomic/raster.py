"""Nodes whose image is made on the CPU and placed on the GPU: Bitmap (an image file), SVG and Text
(rasterized by resvg). Files are resolved relative to the package, and their modification time is
part of the job key, so editing a file recooks."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from trafilatura3d.nodes.base import Input, Node, Output, Param, ParamKind, PortType

COLOR_MODES = ("Color", "Grayscale")


def _resolve(path: str, package) -> dict:
    if not path:
        return {"file": "", "mtime": 0.0}
    from trafilatura3d.project import alias_of, expand
    if alias_of(path):  # alias://path from the project configuration
        try:
            path = expand(path)
        except KeyError:
            return {"file": "", "mtime": 0.0}
    p = Path(path)
    if not p.is_absolute() and package is not None and package.path is not None:
        p = package.path.parent / p
    p = p.resolve()
    return {"file": str(p), "mtime": p.stat().st_mtime if p.exists() else 0.0}


def _read_image(path: str) -> np.ndarray:
    """An image file as float32 RGBA (grayscale spread to RGB, alpha 1 when missing)."""
    import OpenImageIO as oiio
    from trafilatura3d.imageio import read_config
    buf = oiio.ImageBuf(path, 0, 0, read_config())
    if not buf.read(force=True):  # a missing or broken file: an error, not OIIO's assertion in get_pixels
        raise OSError(f"{path}: {buf.geterror()}")
    a = buf.get_pixels(oiio.FLOAT)
    if a is None or buf.has_error:
        raise OSError(f"{path}: {buf.geterror()}")
    h, w, n = a.shape
    out = np.ones((h, w, 4), np.float32)
    out[:, :, :3] = a[:, :, :1] if n < 3 else a[:, :, :3]
    if n in (2, 4):
        out[:, :, 3] = a[:, :, n - 1]
    return out


_RASTERIZERS = None  # the threads each waiting for a resvg process (`_rasterized`)
_RASTERIZING: dict = {}  # upload key -> the render being made (a cook started over takes it up)
# A Python of its own with resvg alone: the options in (pickled), the PNG bytes out. Not multiprocessing: its spawn
# runs the parent's main script again in the child (any script without a __main__ guard).
_RESVG = ("import pickle, sys, resvg_py; "
          "sys.stdout.buffer.write(resvg_py.svg_to_bytes(**pickle.load(sys.stdin.buffer)))")


def _resvg(kwargs: dict) -> bytes:
    """resvg's PNG bytes of an SVG, from a process of its own (this thread waits without Python's lock)."""
    import pickle
    import subprocess
    import sys
    done = subprocess.run([sys.executable, "-I", "-c", _RESVG], input=pickle.dumps(kwargs), capture_output=True)
    if done.returncode:
        raise ValueError(f"SVG: {done.stderr.decode(errors='replace').strip().splitlines()[-1:]}")
    return done.stdout


def _rasterized(ctx, key: str, **kwargs):
    """The texture of an SVG rasterized by resvg (`kwargs` its options), uploaded by `key`: a generator whose step
    waits for the render (`Pending`), its texture the return value. resvg holds Python's lock for its whole render (a
    4096 mask: 5 s), so it renders in a process of its own: the interface answers meanwhile."""
    from trafilatura3d.evaluator import Pending
    global _RASTERIZERS
    data = None
    if not ctx.ev.uploaded(key):
        future = _RASTERIZING.get(key)
        if future is None:
            if _RASTERIZERS is None:
                from concurrent.futures import ThreadPoolExecutor
                _RASTERIZERS = ThreadPoolExecutor(max(1, min(4, (os.cpu_count() or 2) // 2)), "resvg")
            future = _RASTERIZING[key] = _RASTERIZERS.submit(_resvg, kwargs)
            future.add_done_callback(lambda _: _RASTERIZING.pop(key, None))
        yield Pending(future)
        data = future.result()
    return ctx.ev.upload(key, lambda: _png_image(data))


def _png_image(data: bytes) -> np.ndarray:
    """resvg's PNG bytes as float32 RGBA (read back through a temp file)."""
    fd, tmp = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(bytes(data))
        return _read_image(tmp)
    finally:
        os.unlink(tmp)


class Place(Node):
    """Helper: puts an uploaded image at the job's size (stretch, smooth or nearest) and, when asked,
    lays it over the Background input or color by its alpha."""

    type = "place_image"
    registered = False
    inputs = {"image": Input(PortType.COLOR), "background": Input()}
    internal_uniforms = {"filtering": ParamKind.INT, "composite": ParamKind.INT, "background_rgba": ParamKind.COLOR,
                         "tint": ParamKind.COLOR}


PLACE = Place()


def place(ctx, job, image_texture, filtering=1, composite=False, background=(0, 0, 0, 1), tint=(1, 1, 1, 1)):
    ctx.dispatch(PLACE, job, {"filtering": filtering, "composite": int(composite),
                              "background_rgba": list(background), "tint": list(tint)},
                 srcs={"image": (image_texture, False)})


class Bitmap(Node):
    """An image file, resampled to the node's Output Size (by default absolute: the image's own size,
    rounded to powers of two)."""

    type = "bitmap"
    label = "Bitmap"
    category = "Atomic nodes"
    catalog = "bitmap"
    color_mode = "color_mode"
    outputs = {"output": Output()}
    params = {
        "color_mode": Param(ParamKind.ENUM, "Color", "Color mode", choices=COLOR_MODES),
        "pkg_resource_path": Param(ParamKind.STRING, "", "PKG resource path"),
        "resize_method": Param(ParamKind.ENUM, "Smooth stretch", "Resize method",
                               choices=("Smooth stretch", "Nearest stretch")),
    }

    def extra(self, values, package):
        info = _resolve(values["pkg_resource_path"], package)
        if info["file"] and info["mtime"]:
            import OpenImageIO as oiio
            inp = oiio.ImageInput.open(info["file"])
            if inp is not None:
                info["image_size"] = [inp.spec().width, inp.spec().height]
                inp.close()
        return info

    def base_defaults_for(self, values, extra):
        if "image_size" not in extra:
            return {}
        w, h = extra["image_size"]
        log2 = [max(0, min(13, round(math.log2(max(v, 1))))) for v in (w, h)]
        return {"output_size": {"mode": "absolute", "value": log2}}

    def encode(self, ctx, job):
        path = job.extra["file"]
        if path and job.extra["mtime"]:
            tex = ctx.ev.upload(f"bitmap:{path}:{job.extra['mtime']}", lambda: _read_image(path))
        else:  # no image: transparent black, like an empty resource
            tex = ctx.ev.constant((0.0, 0.0, 0.0, 0.0))
        place(ctx, job, tex, filtering=int(job.values["resize_method"] == "Smooth stretch"))


class SVG(Node):
    """An SVG resource rasterized at the node's size over the Background input (or Background color)."""

    type = "svg"
    label = "SVG"
    category = "Atomic nodes"
    catalog = "svg"
    color_mode = "color_mode"
    inputs = {"background": Input(label="Background")}
    outputs = {"output": Output()}
    params = {
        "color_mode": Param(ParamKind.ENUM, "Color", "Color mode", choices=COLOR_MODES),
        "background_color": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 0.0], "Background color"),
        "pkg_resource_path": Param(ParamKind.STRING, "", "PKG resource path"),
    }

    def extra(self, values, package):
        return _resolve(values["pkg_resource_path"], package)

    def encode(self, ctx, job):
        path, (w, h) = job.extra["file"], job.size
        if path and job.extra["mtime"]:
            tex = yield from _rasterized(ctx, f"svg:{path}:{job.extra['mtime']}:{w}x{h}", svg_path=path, width=w,
                                         height=h)
        else:
            tex = ctx.ev.constant((0.0, 0.0, 0.0, 0.0))
        place(ctx, job, tex, composite=True, background=job.values["background_color"])


class Text(Node):
    """Text over the Background input (or Background color), drawn in UV space (it stretches with a non-square
    output). Font size is the font's line height (ascent + descent) as a fraction of the image; Position moves the
    anchor from the image's centre, where the first line's baseline starts (Alignment: the text starts, is centred
    or ends there), the next lines below it; Transformation is a 2x2 matrix mapping the output to the text, about
    the anchor. Font: a font file (.ttf/.otf/.ttc, relative to the package), or a system font as "Family|Style"
    (or "Family"; empty: a default sans-serif)."""

    type = "text"
    label = "Text"
    category = "Atomic nodes"
    catalog = "text"
    color_mode = "color_mode"
    inputs = {"background": Input(label="Background")}
    outputs = {"output": Output()}
    params = {
        "color_mode": Param(ParamKind.ENUM, "Grayscale", "Color mode", choices=("Grayscale", "Color")),
        "text": Param(ParamKind.STRING, "Text", "Text"),
        "font": Param(ParamKind.STRING, "", "Font"),
        "font_size": Param(ParamKind.FLOAT, 0.25, "Font size", 0.0, 1.0),
        "alignment": Param(ParamKind.ENUM, "Center", "Alignment", choices=("Left", "Center", "Right")),
        "transformation": Param(ParamKind.FLOAT4, [1.0, 0.0, 0.0, 1.0], "Transformation", -2.0, 2.0),
        "position": Param(ParamKind.FLOAT2, [0.0, 0.0], "Position", -1.0, 1.0),
        "background": Param(ParamKind.COLOR, [0.0, 0.0, 0.0, 1.0], "Background"),
        "font_color": Param(ParamKind.COLOR, [1.0, 1.0, 1.0, 1.0], "Font color"),
    }

    def extra(self, values, package):
        """The font's file (a file, or the system font of the name: the default one when the system has no such
        font) and face; none when there is no font at all."""
        from trafilatura3d import fonts
        font = values["font"]
        if font.lower().endswith(fonts.FONT_SUFFIXES):
            return {"font_file": _resolve(font, package), "font_face": 0}
        found = fonts.find(*fonts.split_spec(font)) or fonts.find("")
        if found is None:
            return {}
        return {"font_file": _resolve(str(found[0]), None), "font_face": found[1]}

    def encode(self, ctx, job):
        w, h = job.size
        v = job.values
        file = job.extra.get("font_file", {}).get("file", "")
        svg, font_files = _text_svg(v, file, job.extra.get("font_face", 0)) if file and Path(file).is_file() else \
            ('<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>', None)  # no font: no text
        tex = yield from _rasterized(ctx, f"text:{job.key}", svg_string=svg, width=w, height=h, font_files=font_files,
                                     skip_system_fonts=True)
        place(ctx, job, tex, composite=True, background=v["background"], tint=v["font_color"])


UV = 1000.0  # the text's canvas: the image as UV x 1000, stretched to the output
_WEIGHTS = (("thin", 100), ("extralight", 200), ("light", 300), ("medium", 500), ("semibold", 600), ("demi", 600),
            ("extrabold", 800), ("bold", 700), ("black", 900), ("heavy", 900))


def _text_svg(v: dict, file: str, face: int) -> tuple[str, list[str]]:
    """The SVG of a Text node's values with the font of `file` (face `face`), and the font files it uses."""
    from trafilatura3d import fonts
    family, style, (ascent, descent, gap) = fonts.face_info(file, face)
    s = style.lower().replace(" ", "").replace("-", "")
    weight = next((wt for name, wt in _WEIGHTS if name in s), 400)
    italic = ' font-style="italic"' if "italic" in s or "oblique" in s else ""
    em = v["font_size"] * UV / (ascent + descent)
    a, b, c, d = v["transformation"]
    det = a * d - b * c
    if abs(det) < 1e-9:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>', []
    inv = (d / det, -b / det, -c / det, a / det)  # the matrix maps the output to the text
    x, y = (0.5 + v["position"][0]) * UV, (0.5 + v["position"][1]) * UV
    anchor = {"Left": "start", "Center": "middle", "Right": "end"}[v["alignment"]]
    spans = "".join(f'<tspan x="0" dy="{0 if i == 0 else ascent + descent + gap}em">{escape(line) or " "}</tspan>'
                    for i, line in enumerate(v["text"].split("\n")))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {UV} {UV}" preserveAspectRatio="none">'
           f'<g transform="translate({x} {y}) matrix({inv[0]} {inv[1]} {inv[2]} {inv[3]} 0 0)">'
           f'<text font-size="{em}" font-family="{escape(family)}" font-weight="{weight}"{italic} fill="white" '
           f'text-anchor="{anchor}">{spans}</text></g></svg>')
    return svg, [file]
