"""Export a graph's outputs to image files, as the Export outputs dialog does: one
file per enabled Output node, in one Format for all, named by a Pattern of the output's attributes.

The bit depth follows the output's format within what the file format holds:
PNG and TIFF 8 or 16 bit, TGA, JPEG and BMP 8 bit, EXR half or float. With OCIO color management, an output
whose Color space attribute names a color space of the config is converted to it (from an sRGB texture for
color outputs, raw data for grayscale ones, as the project settings say).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trafilatura3d import imageio
from trafilatura3d.compiler import Context, application_context, compile_graph
from trafilatura3d.evaluator import Evaluator
from trafilatura3d.package import Package
from trafilatura3d.project import IMAGE_FORMATS as FORMATS
PATTERN = "$(graph)_$(identifier)"  # ours: the page gives the variables, not the default
VARIABLES = ("graph", "identifier", "description", "label", "user_data", "group", "colorspace")


def default_format() -> str:
    """The project's Default image format (Project settings > General)."""
    from trafilatura3d.project import configuration
    fmt = configuration().setting("general", "image_format")
    return fmt if fmt in FORMATS else "png"


def output_attributes(package: Package, graph_id: str) -> list[dict]:
    """The graph's outputs in order: identifier, description, label, user_data, group, colorspace."""
    g = package.graphs[graph_id]
    out = []
    for node in g.output_nodes():
        v = package.definition(node).values(node)
        out.append({"identifier": v["identifier"], "description": v["description"], "label": v["label"],
                    "user_data": v["user_data"], "group": v["group"], "colorspace": v["color_space"]})
    return out


def file_name(pattern: str, graph_id: str, attributes: dict) -> str:
    """The Pattern with its keywords replaced: $(graph), $(identifier), $(description), $(label),
    $(user_data), $(group), $(colorspace) (the latter only with OCIO color management)."""
    from trafilatura3d import colormanagement as cm
    values = {**attributes, "graph": graph_id}
    if not cm.active():
        values["colorspace"] = ""
    name = pattern
    for key in VARIABLES:
        name = name.replace(f"$({key})", str(values.get(key, "")))
    return name


def export(package: Package, graph_id: str, out_dir: str | Path, evaluator: Evaluator | None = None,
           parent: Context | None = None, values: dict | None = None, *, format: str | None = None,
           pattern: str = PATTERN, outputs: set | None = None, size: tuple | None = None) -> list[Path]:
    """Write the graph's outputs. `parent`: the application context (default: the graph's Parent Size), or
    `size` (log2 width, height): the Batch tab's Output size; `values` override the graph's inputs;
    `outputs`: the identifiers to write (default all); `format` defaults to the project's."""
    from trafilatura3d import colormanagement as cm
    from trafilatura3d import plugins
    package.validate()
    ext = format or default_format()
    if ext not in FORMATS:
        raise ValueError(f"format {ext!r} (formats: {', '.join(FORMATS)})")
    if size is not None:
        parent = application_context(tuple(size))
    ev = evaluator or Evaluator()
    compiled = compile_graph(package, graph_id, parent, evaluator=ev, values=values)
    attributes = {a["identifier"]: a for a in output_attributes(package, graph_id)}
    chosen = {k: v for k, v in compiled.outputs.items() if outputs is None or k in outputs}
    out_dir = Path(out_dir)
    paths = {identifier: out_dir / f"{file_name(pattern, graph_id, attributes.get(identifier, {'identifier': identifier}))}.{ext}"
             for identifier in chosen}
    plugins.emit("before_export", package=package, graph_id=graph_id, out_dir=out_dir,
                 paths=list(paths.values()))  # Veto refuses
    out_dir.mkdir(parents=True, exist_ok=True)
    ev.cook([job for job, _ in chosen.values()])
    written = []
    for identifier, (job, out) in chosen.items():
        pixels = ev.read(job, out)
        if not job.is_gray(out):
            opaque = pixels.shape[2] == 4 and np.all(pixels[:, :, 3] == pixels[:, :, 3].dtype.type(
                255 if pixels.dtype == np.uint8 else 1))
            pixels = pixels[:, :, :3] if opaque or ext in ("jpg", "bmp") else pixels
        target = job.values.get("color_space", "")
        if cm.known(target):
            src = cm.data_space() if job.is_gray(out) else cm.texture_space()
            if src != target:
                was_int = pixels.dtype != np.float32 and pixels.dtype != np.float16
                scale = 255.0 if pixels.dtype == np.uint8 else 65535.0 if pixels.dtype == np.uint16 else 1.0
                f = pixels.astype(np.float32) / scale
                f = f if f.ndim == 3 else f[..., None]
                f = cm.convert(np.repeat(f, 3, -1) if f.shape[-1] == 1 else f, src, target)
                pixels = (f[..., :1] if job.is_gray(out) else f).astype(np.float32)
                if was_int:
                    pixels = np.clip(pixels, 0, 1)
        path = paths[identifier]
        imageio.write(path, convert(pixels, job.depths[out], ext))
        written.append(path)
    plugins.emit("outputs_exported", package=package, graph_id=graph_id, paths=written)
    return written


def convert(pixels: np.ndarray, depth: str, ext: str) -> np.ndarray:
    """Pixels in the type the file gets: integers for PNG/TIFF (8 or 16 bit) and TGA/JPEG/BMP (8 bit),
    half/float for EXR."""
    if ext == "exr":
        return pixels.astype(np.float32 if depth == "32f" else np.float16) if pixels.dtype != np.uint8 \
            else (pixels / 255.0).astype(np.float16)
    if pixels.dtype == np.uint8:
        return pixels
    if ext in ("tga", "jpg", "bmp") or depth == "8":
        return (np.clip(pixels, 0, 1) * 255 + 0.5).astype(np.uint8)
    return (np.clip(pixels, 0, 1) * 65535 + 0.5).astype(np.uint16)
