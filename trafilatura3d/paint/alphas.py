"""The brush alphas (Alpha section of the Paint tool; Assets > Alphas): grayscale stamp shapes, generated in
code (no image files): our library of the
same kinds of shapes (rounds of several hardnesses, geometric shapes, sewing stitches for paths, splatters,
grunges, cracks, scratches, sprays, bristles). Every function returns an n x n array in 0-1, deterministic."""

from __future__ import annotations

import numpy as np


def _grid(n):
    y, x = np.mgrid[0:n, 0:n]
    return (x + 0.5) / n * 2 - 1, (y + 0.5) / n * 2 - 1


def _r(n):
    x, y = _grid(n)
    return np.hypot(x, y)


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def round_(hardness):
    """A disc whose edge falls off over (1 - hardness) of the radius."""
    def f(n):
        r = _r(n)
        inner = min(hardness, 0.999)
        return _smooth((1 - r) / max(1 - inner, 1e-3)) if hardness < 1 else np.clip((1 - r) * n / 4, 0, 1)
    return f


def _value_noise(n, cells, seed):
    rng = np.random.default_rng(seed)
    g = rng.random((cells + 1, cells + 1))
    g[-1, :], g[:, -1] = g[0, :], g[:, 0]
    t = np.linspace(0, cells, n, endpoint=False)
    i = t.astype(int)
    f = _smooth(t - i)
    a = g[i][:, i] * (1 - f)[None, :] + g[i][:, i + 1] * f[None, :]
    b = g[i + 1][:, i] * (1 - f)[None, :] + g[i + 1][:, i + 1] * f[None, :]
    return a * (1 - f)[:, None] + b * f[:, None]


def _fbm(n, seed, octaves=5, base=4):
    out, amp, tot = np.zeros((n, n)), 1.0, 0.0
    for k in range(octaves):
        out += amp * _value_noise(n, base << k, seed + k * 101)
        tot += amp
        amp *= 0.5
    return out / tot


def _falloff(n, hardness=0.3):
    return _smooth((1 - _r(n)) / (1 - hardness))


def grunge(seed, threshold=0.5, contrast=6.0):
    def f(n):
        g = _fbm(n, seed)
        return np.clip((g - threshold) * contrast + 0.5, 0, 1) * _falloff(n)
    return f


def splatter(seed, blobs=14):
    def f(n):
        rng = np.random.default_rng(seed)
        x, y = _grid(n)
        out = np.zeros((n, n))
        for _ in range(blobs):
            cx, cy = rng.normal(0, 0.35, 2)
            rad = rng.uniform(0.04, 0.25)
            out = np.maximum(out, _smooth((rad - np.hypot(x - cx, y - cy)) / (rad * 0.4)))
        edge = _fbm(n, seed + 7, 4, 6)
        return np.clip(out * (0.7 + edge * 0.6), 0, 1) * _falloff(n, 0.6)
    return f


def cracks(seed):
    def f(n):
        g = _fbm(n, seed, 5, 3)
        ridge = 1 - np.abs(g - 0.5) * 2
        return np.clip((ridge - 0.9) * 12, 0, 1) * _falloff(n, 0.5)
    return f


def scratches(seed, count=18):
    def f(n):
        rng = np.random.default_rng(seed)
        x, y = _grid(n)
        out = np.zeros((n, n))
        for _ in range(count):
            a = rng.uniform(-0.4, 0.4) + np.pi / 4
            off = rng.uniform(-0.7, 0.7)
            width = rng.uniform(0.004, 0.015)
            d = np.abs(x * np.sin(a) - y * np.cos(a) - off)
            along = x * np.cos(a) + y * np.sin(a)
            length = rng.uniform(0.3, 0.9)
            out = np.maximum(out, _smooth(1 - d / width) * _smooth((length - np.abs(along - rng.uniform(-0.3, 0.3))) * 8))
        return out * _falloff(n, 0.7)
    return f


def spray(seed, dots=900):
    def f(n):
        rng = np.random.default_rng(seed)
        out = np.zeros((n, n))
        pts = rng.normal(0, 0.38, (dots, 2))
        for px, py in pts:
            ix, iy = int((px * 0.5 + 0.5) * n), int((py * 0.5 + 0.5) * n)
            if 0 <= ix < n and 0 <= iy < n:
                out[iy, ix] = 1.0
        k = max(n // 128, 1)
        if k > 1:
            from numpy.lib.stride_tricks import sliding_window_view
            out = np.pad(out, k // 2)
            out = sliding_window_view(out, (k, k)).max((-1, -2))[:n, :n]
        return out * _falloff(n, 0.2)
    return f


def bristles(seed, count=40):
    def f(n):
        rng = np.random.default_rng(seed)
        x, y = _grid(n)
        out = np.zeros((n, n))
        for _ in range(count):
            cx = rng.uniform(-0.8, 0.8)
            w = rng.uniform(0.01, 0.04)
            out = np.maximum(out, _smooth(1 - np.abs(x - cx) / w) * rng.uniform(0.4, 1.0))
        return out * _smooth((1 - np.abs(y)) * 3) * _smooth((1 - np.abs(x)) * 3)
    return f


def _shape(fn):
    return lambda n: np.clip(fn(*_grid(n), n), 0, 1)


def _polygon(sides, inner=None):
    """A regular polygon (inner: a star with that inner radius ratio)."""
    def f(x, y, n):
        a = np.arctan2(y, x) + np.pi / 2
        r = np.hypot(x, y)
        seg = 2 * np.pi / sides
        if inner is None:
            k = np.cos(np.pi / sides) / np.cos((a % seg) - seg / 2)
        else:
            t = np.abs((a % seg) / seg * 2 - 1)
            k = inner + (1 - inner) * t
        return (0.9 * k - r) * n / 4
    return f


def stitch(angles, half_len=0.7, half_width=0.13):
    """Sewing stitches for the Path tool (ours): a thread segment along X (the path's direction with Follow Path)
    per angle (degrees), rounded at its ends, bulging across like a thread (a relief when painted in Height)."""
    def f(n):
        x, y = _grid(n)
        out = np.zeros((n, n))
        for a in angles:
            c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
            u, v = x * c + y * s, -x * s + y * c
            along = np.clip(np.abs(u) - (half_len - half_width), 0, None)  # past the straight part: the round end
            d = np.hypot(along, v) / half_width  # 0 on the thread's axis, 1 at its edge
            out = np.maximum(out, np.sqrt(np.clip(1 - d * d, 0, 1)))
        return out
    return f


ALPHAS = {
    "Round hard": round_(1.0),
    "Round 75": round_(0.75),
    "Round soft": round_(0.5),
    "Round 25": round_(0.25),
    "Round very soft": round_(0.0),
    "Round blurred": lambda n: np.exp(-(_r(n) * 2.2) ** 2) * (_r(n) < 1),
    "Square": lambda n: np.ones((n, n)),
    "Square soft": _shape(lambda x, y, n: _smooth((0.95 - np.maximum(np.abs(x), np.abs(y))) * 6)),
    "Triangle": _shape(_polygon(3)),
    "Diamond": _shape(lambda x, y, n: (0.95 - np.abs(x) - np.abs(y)) * n / 4),
    "Hexagon": _shape(_polygon(6)),
    "Flower 5": _shape(_polygon(5, 0.45)),
    "Flower 6": _shape(_polygon(6, 0.55)),
    "Ring": lambda n: np.clip(1 - np.abs(_r(n) - 0.8) * 8, 0, 1),
    "Ring thin": lambda n: np.clip(1 - np.abs(_r(n) - 0.85) * 25, 0, 1),
    "Cross": _shape(lambda x, y, n: (0.15 - np.minimum(np.abs(x), np.abs(y))) * n / 4 * (np.maximum(np.abs(x), np.abs(y)) < 0.95)),
    "Line": _shape(lambda x, y, n: _smooth((0.08 - np.abs(y)) * 30) * _smooth((0.95 - np.abs(x)) * 10)),
    "Stitch": stitch((0,)),
    "Stitch slanted": stitch((35,), half_len=0.75),
    "Stitch cross": stitch((40, -40), half_len=0.8, half_width=0.11),
    "Dots": lambda n: np.clip(1 - np.hypot(((_grid(n)[0] + 1) * 3) % 1 - 0.5, ((_grid(n)[1] + 1) * 3) % 1 - 0.5) * 4, 0, 1) * _falloff(n, 0.8),
    **{f"Splatter {i + 1}": splatter(10 + i, 10 + 4 * i) for i in range(4)},
    **{f"Grunge {i + 1}": grunge(30 + i, 0.45 + 0.03 * i) for i in range(5)},
    "Grunge soft": grunge(40, 0.4, 2.5),
    **{f"Cracks {i + 1}": cracks(50 + i) for i in range(3)},
    **{f"Scratches {i + 1}": scratches(60 + i, 12 + 8 * i) for i in range(3)},
    "Spray": spray(70),
    "Spray dense": spray(71, 2500),
    "Bristles": bristles(80),
    "Bristles sparse": bristles(81, 18),
}
DEFAULT_ALPHA = "Round soft"
