"""Color management (Project settings > Color management): Legacy (sRGB, as before) or OpenColorIO.

With OCIO, images keep their values in the graph; the color spaces matter where
images are shown and written:
- the 2D View shows an image through its color space -> the chosen display / view;
- the 3D View renders in the working (scene linear) space and shows it through the display / view
  (the view's own tone mapping, e.g. ACES 2.0);
- export converts an output from its source space to the color space set on its Output node.
OCIO's GPU shaders come in GLSL/HLSL/MSL, not WGSL, so views sample a 3D LUT baked with OCIO's CPU
processor (33^3, with a log2 shaper for scene linear sources)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from trafilatura3d.project import configuration

LUT_SIZE = 33
LOG_MIN, LOG_MAX = -12.0, 8.0  # scene linear shaper range in stops (log2)
BUILTIN = "studio-config-latest"


def mode() -> str:
    return configuration().setting("color_management", "mode")


def active() -> bool:
    return mode() == "OCIO"


@lru_cache(maxsize=4)
def _config(path: str):
    import PyOpenColorIO as ocio
    return ocio.Config.CreateFromFile(path) if path else ocio.Config.CreateFromBuiltinConfig(BUILTIN)


def config():
    return _config(configuration().setting("color_management", "ocio_config") or "")


def color_spaces() -> list[str]:
    return list(config().getColorSpaceNames()) if active() else []


def displays() -> list[str]:
    return list(config().getDisplays()) if active() else []


def views(display: str) -> list[str]:
    return list(config().getViews(display)) if active() else []


def _pick(names, *wanted):
    for w in wanted:
        if w and w in names:
            return w
    return names[0] if names else ""


def working_space() -> str:
    import PyOpenColorIO as ocio
    cfg = config()
    set_ = configuration().setting("color_management", "working_space")
    return set_ if set_ and cfg.getColorSpace(set_) else cfg.getRoleColorSpace(ocio.ROLE_SCENE_LINEAR) or ""


def texture_space() -> str:
    """Where color images live when nothing says otherwise (an sRGB encoded texture)."""
    names = list(config().getColorSpaceNames())
    return _pick(names, configuration().setting("color_management", "texture_space"), "sRGB - Texture",
                 "sRGB Encoded Rec.709 (sRGB)", "srgb_tx", "Utility - sRGB - Texture")


def data_space() -> str:
    """Non color data (grayscale masks, normals, roughness): passed through."""
    names = list(config().getColorSpaceNames())
    return _pick(names, configuration().setting("color_management", "data_space"), "Raw", "raw", "Utility - Raw")


def default_display_view() -> tuple[str, str]:
    cfg = config()
    d = configuration().setting("color_management", "display") or cfg.getDefaultDisplay()
    if d not in cfg.getDisplays():
        d = cfg.getDefaultDisplay()
    v = configuration().setting("color_management", "view") or cfg.getDefaultView(d)
    if v not in cfg.getViews(d):
        v = cfg.getDefaultView(d)
    return d, v


def known(space: str) -> bool:
    return bool(space) and active() and config().getColorSpace(space) is not None


@lru_cache(maxsize=16)
def _lut(config_key: str, src: str, display: str, view: str, log_shaper: bool) -> np.ndarray:
    import PyOpenColorIO as ocio
    cfg = config()
    t = ocio.DisplayViewTransform()
    t.setSrc(src)
    t.setDisplay(display)
    t.setView(view)
    cpu = cfg.getProcessor(t).getDefaultCPUProcessor()
    g = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")  # texture x = r, y = g, z = b
    rgb = np.stack([r, gg, b], -1).reshape(-1, 3).copy()
    if log_shaper:
        rgb = np.exp2(LOG_MIN + rgb * (LOG_MAX - LOG_MIN)).astype(np.float32)
        rgb[np.all(np.stack([r, gg, b], -1).reshape(-1, 3) == 0, axis=1)] = 0.0
    cpu.applyRGB(rgb)
    out = np.ones((LUT_SIZE, LUT_SIZE, LUT_SIZE, 4), np.float32)
    out[..., :3] = np.clip(rgb.reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3), 0.0, 1.0)
    return out


def display_lut(src: str, display: str, view: str, scene_linear: bool = False) -> np.ndarray:
    """(N, N, N, 4) float32: src -> display/view. `scene_linear` sources use a log2 shaper: index
    with t = (log2(x) - LOG_MIN) / (LOG_MAX - LOG_MIN)."""
    key = configuration().setting("color_management", "ocio_config") or BUILTIN
    return _lut(key, src, display, view, scene_linear)


def convert(pixels: np.ndarray, src: str, dst: str) -> np.ndarray:
    """An image (H, W, C float32) from one color space to another (export)."""
    if not (active() and src and dst and src != dst):
        return pixels
    cpu = config().getProcessor(src, dst).getDefaultCPUProcessor()
    out = np.ascontiguousarray(pixels[..., :3], np.float32).copy()
    cpu.applyRGB(out.reshape(-1, 3))
    if pixels.shape[-1] > 3:
        return np.concatenate([out, pixels[..., 3:]], -1)
    return out


def reset_cache() -> None:
    _config.cache_clear()
    _lut.cache_clear()
