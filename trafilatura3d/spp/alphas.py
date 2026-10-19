"""Brush alphas of an imported project that come as compiled procedurals (not read): rebuilt from our own generator
nodes. Each alpha is a grayscale base texture shaped by three parameters, measured on renders of the originals:

- `invert`: the base becomes 1 - base;
- `hardness` h: levels on the base, input black at 1 - 2h below one half, input white at 2 (1 - h) above it;
- `shape` s: a square falloff M = S((1 - |x|) / s) S((1 - |y|) / s) (S: smoothstep, x and y in -1..1) subtracted
  as clamp(base + M - 1), before or after the levels depending on the alpha.

The base is one of our generators (cooked at `size_log2` and resized to 512: smaller makes its lines thicker),
optionally modulated by a second one (a * (keep + (1 - keep) * b)), its values remapped to the original's
distribution (17 quantiles, measured).

Two more are drawn here, from their measured geometry: a soft disc (radius half the frame, an edge 1 - hardness wide
centered on it) and a scatter of dots (diameter 0.3 x size of the frame, uniform in the square and kept within
`spawn` of the center, opacity 1 - random opacity x U, diameter x 2^(3.5 random size (2U - 1)))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE_LOG2 = 9


@dataclass(frozen=True)
class Base:
    node: str  # our generator
    invert: bool  # its values turned over before the remap
    quantiles: tuple  # the original base's values at 0, 1/16 .. 1 of its distribution
    levels_first: bool = True  # the levels on the base, then the falloff subtracted (False: the other way round)
    params: tuple = ()  # (name, value) of the generator
    size_log2: int = SIZE_LOG2
    modulator: str | None = None  # a second generator multiplied in
    keep: float = 0.0  # how much of the base the modulator leaves where it is black


BASES = {
    "alpha_dirt_02": Base("grunge_stains_legacy", False, (
        0.012, 0.385, 0.467, 0.519, 0.555, 0.587, 0.610, 0.634, 0.655, 0.677, 0.694, 0.715, 0.736, 0.756, 0.778,
        0.808, 0.996), levels_first=False),
    "alpha_dirt_03": Base("grunge_fibrous", False, (
        0.015, 0.464, 0.531, 0.575, 0.610, 0.639, 0.666, 0.692, 0.717, 0.741, 0.767, 0.793, 0.822, 0.853, 0.886,
        0.926, 0.976), levels_first=False),
    "alpha_dirt_04": Base("grunge_blobs", False, (
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.009, 0.066, 0.179, 0.352, 0.496, 0.524,
        0.829, 1.000)),
    "alpha_dirt_06": Base("bnw_noise_rough", True, (
        0.000, 0.000, 0.003, 0.016, 0.038, 0.065, 0.098, 0.133, 0.171, 0.212, 0.256, 0.305, 0.360, 0.426, 0.509,
        0.620, 0.940), modulator="clouds_soft"),
    "alpha_cracks_01": Base("cells_walls", True, (
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.010, 0.050, 0.120, 0.223, 0.342, 0.463, 0.573, 0.663,
        0.738, 0.999), params=(("scale", 18), ("hardness", 0.0)), size_log2=8, modulator="bnw_noise_rough",
        keep=0.3),
    "alpha_cracks_02": Base("crystal_facets", False, (
        0.012, 0.170, 0.208, 0.237, 0.261, 0.283, 0.304, 0.325, 0.346, 0.368, 0.391, 0.416, 0.443, 0.475, 0.516,
        0.575, 1.000)),
}


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def levels(base: np.ndarray, hardness: float) -> np.ndarray:
    h = float(np.clip(hardness, 0.0, 1.0))
    if h <= 0.5:
        return np.clip((base - (1 - 2 * h)) / max(2 * h, 1e-6), 0.0, 1.0)
    return np.clip(base / max(2 * (1 - h), 1e-6), 0.0, 1.0)


def falloff(n: int, shape: float) -> np.ndarray:
    if shape <= 0:
        return np.ones((n, n))
    t = (np.arange(n) + 0.5) / n * 2 - 1
    m = _smoothstep((1 - np.abs(t)) / shape)
    return m[:, None] * m[None, :]


def shaped(base: np.ndarray, hardness: float, shape: float, invert: bool, levels_first: bool = True) -> np.ndarray:
    """The alpha from its base (n x n in 0-1) and parameters."""
    b = 1 - base if invert else base
    m = falloff(b.shape[0], float(shape))
    if levels_first:
        return np.clip(levels(b, hardness) + m - 1, 0.0, 1.0)
    return levels(np.clip(b + m - 1, 0.0, 1.0), hardness)


def _cook(ev, node_type: str, params: tuple, size_log2: int, seed: int) -> np.ndarray:
    from trafilatura3d.compiler import ABSOLUTE, NODE_DEFAULTS, compile_graph
    from trafilatura3d.package import Package
    pkg = Package()
    g = pkg.new_graph("alpha", size_log2)
    node = g.add_node(node_type)
    for name, value in params:
        g.set_param(node, name, value)
    g.set_base(node, "seed", dict(NODE_DEFAULTS["seed"], value=seed))
    g.set_base(node, "output_format", {"mode": ABSOLUTE, "value": "16f"})
    job = compile_graph(pkg, "alpha").jobs[node]
    a = np.asarray(ev.read(job, next(iter(job.node.outputs))))
    a = a.astype(np.float32) / np.iinfo(a.dtype).max if a.dtype.kind in "ui" else a.astype(np.float32)
    a = a[..., 0] if a.ndim == 3 else a
    n = 1 << SIZE_LOG2
    if a.shape[0] != n:  # bilinear, texel centers aligned
        t = (np.arange(n) + 0.5) * a.shape[0] / n - 0.5
        i = np.clip(np.floor(t).astype(int), 0, a.shape[0] - 2)
        f = np.clip(t - i, 0.0, 1.0)
        a = a[i] * (1 - f)[:, None] + a[i + 1] * f[:, None]
        a = a[:, i] * (1 - f)[None, :] + a[:, i + 1] * f[None, :]
    return a.astype(np.float64)


def base_image(ev, name: str, seed: int = 0) -> np.ndarray:
    """The base of `name` (a BASES key): our generators cooked and remapped to the original's distribution."""
    spec = BASES[name]
    a = _cook(ev, spec.node, spec.params, spec.size_log2, seed)
    if spec.invert:
        a = 1 - a
    if spec.modulator:
        a = a * (spec.keep + (1 - spec.keep) * _cook(ev, spec.modulator, (), SIZE_LOG2, seed))
    ours = np.quantile(a, np.linspace(0, 1, len(spec.quantiles)))
    ours = ours + np.arange(len(ours)) * 1e-9  # increasing, for the interpolation
    return np.interp(a, ours, spec.quantiles)


def disc(hardness: float, n: int = 1 << SIZE_LOG2) -> np.ndarray:
    w = max(1.0 - float(np.clip(hardness, 0.0, 1.0)), 2.0 / n)
    t = (np.arange(n) + 0.5) / n * 2 - 1
    r = np.hypot(t[:, None], t[None, :])
    return _smoothstep((0.5 + w / 2 - r) / w)


def dots(tweaks: dict, seed: int = 0, n: int = 1 << SIZE_LOG2) -> np.ndarray:
    """The dots of up to two layers (`layer-01_*`, `layer-02_*`), the brightest where they overlap."""
    rng = np.random.default_rng(seed)
    spawn = float(tweaks.get("spawn", 0.5))
    t = (np.arange(n) + 0.5) / n * 2 - 1
    out = np.zeros((n, n))
    for layer in ("layer-01", "layer-02"):
        count = int(tweaks.get(f"{layer}_amount", 0))
        pos = rng.uniform(-1, 1, (count, 2))
        u_size, u_opacity = rng.random(count), rng.random(count)
        size, opacity = float(tweaks.get(f"{layer}_size", 0.2)), float(tweaks.get(f"{layer}_opacity", 1.0))
        jitter_size = float(tweaks.get(f"{layer}_random-size", 0.0))
        jitter_opacity = float(tweaks.get(f"{layer}_random_opacity", 0.0))
        for (x, y), us, uo in zip(pos, u_size, u_opacity):
            if np.hypot(x, y) > spawn:
                continue
            r = 0.3 * size * 2.0 ** (3.5 * jitter_size * (2 * us - 1))  # radius in -1..1 units (half the diameter)
            d = np.hypot(t[None, :] - x, t[:, None] - y)
            out = np.maximum(out, np.clip((r - d) * n / 2 + 0.5, 0.0, 1.0) * opacity * (1 - jitter_opacity * uo))
    return out


NAMES = frozenset(BASES) | {"shape_inky", "dots_02"}


def alpha(ev, name: str, tweaks: dict, seed: int = 0) -> np.ndarray:
    """The brush alpha `name` (in NAMES) with the parameter values of the project."""
    if name == "shape_inky":
        return disc(float(tweaks.get("hardness", 0.5))).astype(np.float32)
    if name == "dots_02":
        return dots(tweaks, seed).astype(np.float32)
    spec = BASES[name]
    return shaped(base_image(ev, name, seed), float(tweaks.get("hardness", 0.5)), float(tweaks.get("shape", 0.75)),
                  bool(tweaks.get("invert", 0)), spec.levels_first).astype(np.float32)
