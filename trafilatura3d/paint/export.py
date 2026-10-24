"""Export textures (Export window documentation: Export settings, Output templates, List of exports).

An output template lists output maps: a file name with keywords ($project, $mesh, $textureSet, $udim,
$colorSpace; "/" makes folders), a map type (Gray, RGB, R+G+B, RGB+A, R+G+B+A) whose slots take a component
of an input (a Texture Set channel, a mesh map or a converted map), a file format and a bit depth. Channels a
Texture Set lacks export as their default value (the documentation: "a neutral default color per channel").

Ours where the documentation is silent: the file names of the predefined templates, the default padding
(Dilation infinite), the default dilation distance (16 px), the template file format (JSON). Formats are those
the documentation lists that OpenImageIO writes; psd, jpeg-xr, jng, wbmp, xpm and pbm are not offered.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trafilatura3d import imageio
from trafilatura3d.paint.channels import CHANNELS, FORMATS, default_rgba
from trafilatura3d.paint.model import MESH_MAPS, TextureSet

# File type -> (extension, bit depths offered) from the Export settings page, writable formats only.
FILE_TYPES = {
    "bmp": ("bmp", ("8", "8 + dithering")),
    "exr": ("exr", ("16 (floating)", "32 (floating)")),
    "gif": ("gif", ("8", "8 + dithering")),
    "hdr": ("hdr", ("32 (floating)",)),
    "ico": ("ico", ("8", "8 + dithering")),
    "j2k": ("j2k", ("8", "8 + dithering", "16")),
    "jp2": ("jp2", ("8", "8 + dithering", "16")),
    "jpeg": ("jpg", ("8", "8 + dithering")),
    "pfm": ("pfm", ("32 (floating)",)),
    "pgm": ("pgm", ("8", "8 + dithering", "16")),
    "png": ("png", ("8", "8 + dithering", "16")),
    "ppm": ("ppm", ("8", "8 + dithering", "16")),
    "targa": ("tga", ("8", "8 + dithering")),
    "tiff": ("tif", ("8", "8 + dithering", "16", "32 (floating)")),
    "webp": ("webp", ("8", "8 + dithering")),
}
SIZES = ("Based on each Texture Set's size", "128", "256", "512", "1024", "2048", "4096", "8192")
PADDING = ("No padding (passthrough)", "Dilation infinite", "Dilation + transparent",
           "Dilation + default background color", "Dilation + diffusion")
MAP_TYPES = {"Gray": ("L",), "RGB": ("RGB",), "R+G+B": ("R", "G", "B"), "RGB+A": ("RGB", "A"),
             "R+G+B+A": ("R", "G", "B", "A")}
COMPONENTS = ("RGB", "R", "G", "B", "Alpha", "Grayscale")
# Converted maps (Output templates page) not deprecated.
CONVERTED = ("Normal OpenGL", "Normal DirectX", "Mixed AO", "Diffuse", "Specular", "Glossiness", "Reflection",
             "1/ior", "Glossiness2", "f0")
DEFAULT_DILATION = 16


@dataclass
class OutputMap:
    name: str  # file name with keywords, no extension
    type: str  # MAP_TYPES key
    slots: dict = field(default_factory=dict)  # slot (L, RGB, R, G, B, A) -> {"input": "channel:basecolor", "component": "RGB"}
    file_type: str = "png"
    depth: str = "8"


@dataclass
class Template:
    name: str
    maps: list = field(default_factory=list)
    predefined: bool = False  # generated, not editable

    def to_dict(self):
        return {"name": self.name, "maps": [vars(m) for m in self.maps]}

    @staticmethod
    def from_dict(d) -> Template:
        return Template(d["name"], [OutputMap(**m) for m in d["maps"]])


def _doc_channels(alpha: bool) -> Template:
    """Document channels + Normal + AO: every channel of the Texture Set, the converted normal and the mixed AO
    (the predefined template's name says what it holds; the file names are ours)."""
    t = Template(f"Document channels + Normal + AO ({'With' if alpha else 'No'} Alpha)", predefined=True)
    t.maps.append(OutputMap("$mesh_$textureSet_document", "RGB+A" if alpha else "RGB", {}))  # expanded per set
    return t


PREDEFINED = [_doc_channels(False), _doc_channels(True), Template("Mesh Maps", [OutputMap("$mesh_$textureSet_mesh_maps", "RGB")],
                                                                  predefined=True)]


def expand(template: Template, ts: TextureSet) -> list[OutputMap]:
    """The output maps of a template for one Texture Set (the predefined ones depend on its channels)."""
    if template.name.startswith("Document channels"):
        alpha = "With Alpha" in template.name
        maps = []
        for ch in ts.channel_ids():
            if ch in ("normal", "ambientocclusion"):
                continue
            label = CHANNELS[ch].label.title().replace(" ", "")
            gray = FORMATS[ts.format_of(ch)][0]
            if alpha:
                slots = {"RGB": {"input": f"channel:{ch}", "component": "RGB"},
                         "A": {"input": f"channel:{ch}", "component": "Alpha"}}
                maps.append(OutputMap(f"$mesh_$textureSet_{label}", "RGB+A", slots))
            else:
                maps.append(OutputMap(f"$mesh_$textureSet_{label}", "Gray" if gray else "RGB",
                                      {("L" if gray else "RGB"): {"input": f"channel:{ch}",
                                                                  "component": "Grayscale" if gray else "RGB"}}))
        maps.append(OutputMap("$mesh_$textureSet_Normal", "RGB", {"RGB": {"input": "converted:Normal OpenGL", "component": "RGB"}}))
        maps.append(OutputMap("$mesh_$textureSet_AO", "Gray", {"L": {"input": "converted:Mixed AO", "component": "Grayscale"}}))
        return maps
    if template.name == "Mesh Maps" and template.predefined:
        return [OutputMap(f"$mesh_$textureSet_{suffix}", "RGB", {"RGB": {"input": f"mesh:{slot}", "component": "RGB"}},
                          depth="16")
                for slot, _label, suffix in MESH_MAPS if slot in ts.mesh_maps]
    return copy.deepcopy(template.maps)


def file_name(pattern: str, project_name: str, mesh: str, ts: TextureSet, udim: int, uv_tiles: bool = False,
              color_space: str = "") -> str:
    """A map's file name from its pattern; with the UV Tile workflow a pattern without $udim gets ".<UDIM>" (ours,
    Mari's convention: the tiles would overwrite each other)."""
    out = pattern
    if uv_tiles and "$udim" not in out:
        out += ".$udim"
    for k, v in (("$project", project_name), ("$mesh", Path(mesh).stem), ("$textureSet", ts.name),
                 ("$textureset", ts.name), ("$udim", str(udim)), ("$colorSpace", color_space)):
        out = out.replace(k, v)
    return out


@dataclass
class ExportSettings:
    output_dir: str = ""
    template: str = "Document channels + Normal + AO (No Alpha)"
    file_type: str = "png"  # or "Based on output template"
    depth: str = "8"
    size: str = SIZES[0]
    padding: str = "Dilation infinite"
    dilation: int = DEFAULT_DILATION
    texture_sets: list = field(default_factory=list)  # names to export ([] = all)


class Exporter:
    """Reads the session's stack results and writes the files of a template."""

    def __init__(self, session):
        self.s = session

    def _channel(self, ts, udim, ch, padding, dilation) -> np.ndarray:
        """A channel of a tile as float RGBA HxWx4, padded; the default value where the Texture Set has no such
        channel."""
        sc = self.s.compiler(udim)
        h, w = ts.size_of(udim)[1], ts.size_of(udim)[0]
        if ch not in ts.channel_ids():
            d = default_rgba(ch)
            out = np.empty((h, w, 4), np.float32)
            out[:] = (*d[:3], 1.0)
            return out
        return self._read(ts, udim, sc.channel(ts, ch), padding, dilation, default_rgba(ch))

    def _read(self, ts, udim, job, padding, dilation, default=(0, 0, 0, 0)) -> np.ndarray:
        sc, ev = self.s.compiler(udim), self.s.ev
        self.s.uv_mask(ts, udim)  # the UV mask the padding reads
        if padding != PADDING[0]:
            distance = -1.0 if padding == "Dilation infinite" else float(dilation)
            outside = 1 if padding == "Dilation + default background color" else 0
            job = sc.padded(ts, job, distance, outside, default)
        a = _float(ev.read(job, "output"))
        if a.shape[:2] != (ts.size_of(udim)[1], ts.size_of(udim)[0]):  # a uniform channel (1x1 job): the whole texture
            a = np.broadcast_to(a[:1, :1], (ts.size_of(udim)[1], ts.size_of(udim)[0], a.shape[2])).copy()
        if padding == "Dilation + diffusion":
            # The texels within the dilation distance of an island are kept, the rest gets the diffused
            # (push-pull) colors of the borders; alpha stays the coverage inside, 0 outside.
            from trafilatura3d.bake import _diffuse
            uv_mask = self.s.uv_masks[(ts.name, udim)]
            mask = sc._texture_job(uv_mask, ("uvmask", ts.name, id(uv_mask)), ts.size_of(udim), "8")
            filled = _float(ev.read(sc.padded(ts, mask, float(dilation), 0), "output"))[..., 3] > 0.5
            tmp = a.copy()
            tmp[..., 3] = filled
            out = _diffuse(tmp)
            out[..., 3] = np.where(filled, a[..., 3], 0.0)
            a = out
        return a

    def input(self, ts, udim, source: str, padding, dilation) -> np.ndarray:
        """An input map of a tile as float RGBA: channel:<id>, mesh:<slot> or converted:<name>."""
        kind, name = source.split(":", 1)
        if kind == "channel":
            return self._channel(ts, udim, name, padding, dilation)
        if kind == "mesh":
            job = self.s.compiler(udim).mesh_map(ts, name)
            if job is None:
                return np.zeros((ts.size_of(udim)[1], ts.size_of(udim)[0], 4), np.float32)
            return self._read(ts, udim, job, padding, dilation)
        return self._converted(ts, udim, name, padding, dilation)

    def _converted(self, ts, udim, name, padding, dilation) -> np.ndarray:
        sc = self.s.compiler(udim)
        if name in ("Normal OpenGL", "Normal DirectX"):
            return self._read(ts, udim, sc.normal(ts, directx_out=name.endswith("DirectX")), padding, dilation,
                              (0.5, 0.5, 1.0, 0.0))
        if name == "Mixed AO":
            job = sc.ambient_occlusion(ts)
            if job is None:
                return np.ones((ts.size_of(udim)[1], ts.size_of(udim)[0], 4), np.float32)
            return self._read(ts, udim, job, padding, dilation, (1, 1, 1, 0))
        base = _linear(self._channel(ts, udim, "basecolor", padding, dilation))
        metal = self._channel(ts, udim, "metallic", padding, dilation)[..., :1]
        rough = self._channel(ts, udim, "roughness", padding, dilation)[..., :1]
        alpha = np.ones_like(metal)
        if name == "Diffuse":
            rgb = _srgb(base[..., :3] * (1 - metal))
        elif name == "Specular":
            rgb = _srgb(0.04 * (1 - metal) + base[..., :3] * metal)
        elif name == "f0":
            rgb = np.repeat(0.04 * (1 - metal) + metal, 3, 2)
        elif name == "Glossiness":
            rgb = np.repeat(1 - rough, 3, 2)
        elif name == "Glossiness2":
            rgb = np.repeat((1 - rough) ** 2, 3, 2)
        elif name == "Reflection":  # white where dielectric (the documentation's description)
            rgb = np.repeat(1 - metal, 3, 2)
        elif name == "1/ior":  # IOR 1.4 dielectric, 100 metal
            rgb = np.repeat((1 / 1.4) * (1 - metal) + (1 / 100) * metal, 3, 2)
        else:
            raise ValueError(f"unknown converted map {name!r}")
        return np.concatenate([rgb, alpha], 2).astype(np.float32)

    def pack(self, ts, udim, m: OutputMap, padding, dilation) -> np.ndarray:
        """The pixels of an output map in a tile: HxWxC float (C = 1, 3 or 4)."""
        cache = {}

        def comp(spec, wanted):
            src = spec.get("input")
            if not src:
                return None
            if src not in cache:
                cache[src] = self.input(ts, udim, src, padding, dilation)
            a = cache[src]
            c = spec.get("component", "RGB")
            if c == "RGB":
                return a[..., :3]
            if c == "Alpha":
                return a[..., 3:4]
            if c == "Grayscale":
                return (a[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
            return a[..., "RGB".index(c[0]):"RGB".index(c[0]) + 1]

        h, w = ts.size_of(udim)[1], ts.size_of(udim)[0]
        parts = []
        for slot in MAP_TYPES[m.type]:
            n = 3 if slot == "RGB" else 1
            v = comp(m.slots.get(slot, {}), slot)
            if v is None:
                v = np.zeros((h, w, n), np.float32)
            elif v.shape[2] != n:
                v = np.repeat(v, n, 2) if v.shape[2] == 1 else v.mean(2, keepdims=True)
            parts.append(v)
        return np.concatenate(parts, 2)

    def run(self, template: Template, settings: ExportSettings, log=None) -> list[Path]:
        """Export the checked Texture Sets; `log(text, error)` reports each file (List of exports)."""
        from trafilatura3d import plugins
        out_dir = Path(settings.output_dir)
        written = []
        sets = [ts for ts in self.s.project.texture_sets if not settings.texture_sets or ts.name in settings.texture_sets]
        work = []
        for ts in sets:
            for m in expand(template, ts):
                for udim in ts.tiles:
                    ftype = m.file_type if settings.file_type == "Based on output template" else settings.file_type
                    depth = m.depth if settings.file_type == "Based on output template" else settings.depth
                    ext, depths = FILE_TYPES[ftype]
                    if depth not in depths:
                        depth = depths[0]
                    name = file_name(m.name, self.s.path.stem if self.s.path else "project", self.s.project.mesh, ts,
                                     udim, bool(self.s.project.uv_tiles))
                    work.append((ts, udim, m, ext, depth, out_dir / f"{name}.{ext}"))
        plugins.emit("before_export_textures", session=self.s, out_dir=out_dir,
                     paths=[w[-1] for w in work])  # Veto refuses
        for ts, udim, m, ext, depth, path in work:
            try:
                pixels = self.pack(ts, udim, m, settings.padding, settings.dilation)
                if settings.size != SIZES[0]:
                    from trafilatura3d.paint.paint import _resample
                    n = int(settings.size)
                    pixels = _resample(pixels, n, n)
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.write(path, _encode(pixels, ext, depth))
                written.append(path)
                if log:
                    log(str(path), False)
            except (OSError, ValueError) as e:
                if log:
                    log(f"{path.name}: {e}", True)
        plugins.emit("textures_exported", session=self.s, paths=written)
        return written


def _float(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / 255.0 if raw.dtype == np.uint8 else raw.astype(np.float32)


def _linear(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    c = out[..., :3]
    out[..., :3] = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return out


def _srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055).astype(np.float32)


def _encode(pixels: np.ndarray, ext: str, depth: str) -> np.ndarray:
    """Pixels in the file's type: 8/16 bit integers (dithered on request) or half/float."""
    if ext in ("pgm",) and pixels.shape[2] != 1:
        pixels = pixels[..., :3].mean(2, keepdims=True)
    if ext in ("ppm", "jpg", "hdr") and pixels.shape[2] == 4:
        pixels = pixels[..., :3]
    if ext in ("ppm", "hdr", "pfm") and pixels.shape[2] == 1 and ext != "pfm":
        pixels = np.repeat(pixels, 3, 2)
    if ext == "pfm" and pixels.shape[2] not in (1, 3):
        pixels = pixels[..., :3]
    if depth.startswith("32"):
        return pixels.astype(np.float32)
    if depth.startswith("16 (floating)"):
        return pixels.astype(np.float16)
    if depth == "16":
        return (np.clip(pixels, 0, 1) * 65535 + 0.5).astype(np.uint16)
    v = np.clip(pixels, 0, 1) * 255
    if depth == "8 + dithering":  # triangular noise of one level before rounding
        rng = np.random.default_rng(0)
        v = v + rng.random(v.shape, np.float32) - rng.random(v.shape, np.float32)
    return np.clip(v + 0.5, 0, 255).astype(np.uint8)


def load_templates(path: Path) -> list[Template]:
    if not path.exists():
        return []
    return [Template.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def library_templates() -> list[Template]:
    """The output templates of the project's libraries (their `export_presets` folder: .json files in the format of
    `save_templates`), read only like the predefined ones."""
    from trafilatura3d.project import configuration
    out = []
    for folder in configuration().library_folders("export_presets"):
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            out += [Template(t.name, t.maps, predefined=True) for t in load_templates(path)]
    return out


def save_templates(path: Path, templates: list[Template]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([t.to_dict() for t in templates if not t.predefined], indent=1), encoding="utf-8")
