"""Primitive meshes for the 3D View with tileable UVs (standard and extended primitives):
positions, normals, tangents (xyz + handedness of the bitangent), UVs, triangle indices, as numpy
arrays."""

from __future__ import annotations

import math

import numpy as np

from trafilatura3d.meshio import handedness as _handedness
from trafilatura3d.meshio import wind as _wind


def _grid(nu: int, nv: int, fn):
    """A (nu+1) x (nv+1) vertex grid from fn(u, v) -> position, normal, tangent; uv = (u, v)."""
    u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1))
    pos, nrm, tan = fn(u, v)
    uv = np.stack([u, v], -1)
    i = np.arange((nu + 1) * (nv + 1)).reshape(nv + 1, nu + 1)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    tris = np.stack([a, c, b, b, c, d], -1).reshape(-1, 3)
    return (pos.reshape(-1, 3), nrm.reshape(-1, 3), tan.reshape(-1, 3), uv.reshape(-1, 2), tris)


def _merge(parts):
    pos, nrm, tan, uv, idx, base = [], [], [], [], [], 0
    for p, n, t, u, i in parts:
        pos.append(p); nrm.append(n); tan.append(t[:, :3]); uv.append(u); idx.append(i + base)
        base += len(p)
    pos, nrm, tan, uv = (np.concatenate(x).astype(np.float32) for x in (pos, nrm, tan, uv))
    idx = _wind(pos, nrm, np.concatenate(idx).astype(np.uint32))
    return pos, nrm, np.concatenate([tan, _handedness(pos, nrm, tan, uv, idx)[:, None]], 1), uv, idx


def plane(segments=128, size=2.0):
    def fn(u, v):
        pos = np.stack([(u - 0.5) * size, np.zeros_like(u), (v - 0.5) * size], -1)
        return pos, np.broadcast_to([0, 1, 0], pos.shape), np.broadcast_to([1, 0, 0], pos.shape)
    return _merge([_grid(segments, segments, fn)])


def sphere(seg=128, tiles=1.0):
    def fn(u, v):
        th, ph = u * 2 * math.pi, v * math.pi
        n = np.stack([np.sin(ph) * np.cos(th), np.cos(ph), np.sin(ph) * np.sin(th)], -1)
        t = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], -1)
        return n, n, t
    p, n, t, uv, i = _grid(seg, seg // 2, fn)
    return _merge([(p, n, t, uv * [2 * tiles, tiles], i)])


def cylinder(seg=128):
    def side(u, v):
        th = u * 2 * math.pi
        n = np.stack([np.cos(th), np.zeros_like(th), np.sin(th)], -1)
        pos = n * 0.8 + np.stack([np.zeros_like(v), 1 - 2 * v, np.zeros_like(v)], -1)
        return pos, n, np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], -1)

    def cap(y, flip):
        def fn(u, v):
            r, th = u * 0.8, v * 2 * math.pi
            pos = np.stack([r * np.cos(th), np.full_like(r, y), r * np.sin(th)], -1)
            return pos, np.broadcast_to([0, 1 if y > 0 else -1, 0], pos.shape), np.broadcast_to([1, 0, 0], pos.shape)
        p, n, t, uv, i = _grid(8, seg, fn)
        uv = p[:, [0, 2]] * 0.5 + 0.5
        return p, n, t, uv, (i[:, ::-1] if flip else i)
    s = _grid(seg, 32, side)
    return _merge([(s[0], s[1], s[2], s[3] * [3, 1], s[4]), cap(1.0, True), cap(-1.0, False)])


CUBE_FACES = [((1, 0, 0), (0, 0, -1)), ((-1, 0, 0), (0, 0, 1)), ((0, 1, 0), (1, 0, 0)),
              ((0, -1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 0)), ((0, 0, -1), (-1, 0, 0))]


def _box_faces(faces, seg, half, round_radius=0.0, inward=False):
    """Faces of a box of half size `half` (one UV square each); `inward` turns the normals inside."""
    parts = []
    for n, t in faces:
        n, t = np.array(n, float), np.array(t, float)
        b = np.cross(n, t)

        def fn(u, v, n=n, t=t, b=b):
            p = n + (u[..., None] - 0.5) * 2 * t - (v[..., None] - 0.5) * 2 * b
            if round_radius > 0:
                inner = np.clip(p, -1 + round_radius, 1 - round_radius)
                d = p - inner
                dn = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
                p = inner + dn * round_radius
                nn = np.where(np.linalg.norm(d, axis=-1, keepdims=True) > 1e-6, dn, n)
            else:
                nn = np.broadcast_to(n, p.shape)
            tt = t - nn * np.sum(nn * t, -1, keepdims=True)
            tt /= np.maximum(np.linalg.norm(tt, axis=-1, keepdims=True), 1e-9)
            return p * half, (-nn if inward else nn), tt
        parts.append(_grid(seg, seg, fn))
    return parts


def cube(seg=64, round_radius=0.0):
    """A cube of side 1.6 (rounded when round_radius > 0), one UV square per face."""
    return _merge(_box_faces(CUBE_FACES, seg, 0.8, round_radius))


def hollow_box(seg=48, wall=0.08):
    """A box open at the top with walls of some thickness: outside, inside and the rim."""
    sides = [f for f in CUBE_FACES if f[0] != (0, 1, 0)]
    parts = _box_faces(sides, seg, 0.8) + _box_faces(sides, seg, 0.8 - wall, inward=True)
    lo, hi = 0.8 - wall, 0.8
    for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # the rim: four trapezoids at the top
        def fn(u, v, sx=sx, sz=sz):
            r = lo + (hi - lo) * v
            along = (u - 0.5) * 2 * r
            x = np.where(sx != 0, sx * r, along)
            z = np.where(sz != 0, sz * r, along)
            pos = np.stack([x, np.full_like(u, 0.8), z], -1)
            return pos, np.broadcast_to([0, 1, 0], pos.shape), np.broadcast_to([1, 0, 0], pos.shape)
        p, n, t, uv, i = _grid(seg, 2, fn)
        parts.append((p, n, t, p[:, [0, 2]] * 0.5 + 0.5, i))
    return _merge(parts)


def inner_box(seg=48):
    """The inside of a box open towards the camera (+Z): a small room."""
    return _merge(_box_faces([f for f in CUBE_FACES if f[0] != (0, 0, 1)], seg, 1.0, inward=True))


def _revolve(profile, seg, u_tiles=3.0):
    """A surface of revolution around Y from a 2D profile [(r, y), ...] (bottom to top)."""
    pr = np.asarray(profile, float)
    seglen = np.linalg.norm(np.diff(pr, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seglen)]) / max(seglen.sum(), 1e-9)
    d = np.gradient(pr, axis=0)
    pn = np.stack([d[:, 1], -d[:, 0]], -1)  # profile normal (outwards for a bottom-to-top profile)
    pn /= np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-12)
    nv = len(pr) - 1

    def fn(u, v):
        th = u * 2 * math.pi
        k = np.clip(np.searchsorted(s, v, side="right") - 1, 0, nv - 1)
        f = np.clip((v - s[k]) / np.maximum(s[k + 1] - s[k], 1e-9), 0, 1)
        r = pr[k, 0] + (pr[k + 1, 0] - pr[k, 0]) * f
        y = pr[k, 1] + (pr[k + 1, 1] - pr[k, 1]) * f
        nr = pn[k, 0] + (pn[k + 1, 0] - pn[k, 0]) * f
        ny = pn[k, 1] + (pn[k + 1, 1] - pn[k, 1]) * f
        c, sn = np.cos(th), np.sin(th)
        pos = np.stack([r * c, y, r * sn], -1)
        n = np.stack([nr * c, ny, nr * sn], -1)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        return pos, n, np.stack([-sn, np.zeros_like(sn), c], -1)
    p, n, t, uv, i = _grid(seg, max(64, len(pr)), fn)
    uv = np.column_stack([uv[:, 0] * u_tiles, 1.0 - uv[:, 1]])
    return p, n, t, uv, i


def rounded_cylinder(seg=128, radius=0.8, half=1.0, round_radius=0.2):
    rr, prof = round_radius, [(0.0, -half)]
    for a in np.linspace(-math.pi / 2, 0, 12):
        prof.append((radius - rr + rr * math.cos(a), -half + rr + rr * math.sin(a)))
    for a in np.linspace(0, math.pi / 2, 12):
        prof.append((radius - rr + rr * math.cos(a), half - rr + rr * math.sin(a)))
    prof.append((0.0, half))
    return _merge([_revolve(prof, seg)])


def cloth(seg=256, size=2.0):
    """A draped cloth: a plane with soft folds."""
    def fn(u, v):
        x, z = (u - 0.5) * size, (v - 0.5) * size
        y = 0.12 * np.sin(3.1 * x + 0.6) * np.cos(2.3 * z) + 0.06 * np.sin(5.3 * z + 1.7 * x)
        dx = 0.12 * 3.1 * np.cos(3.1 * x + 0.6) * np.cos(2.3 * z) + 0.06 * 1.7 * np.cos(5.3 * z + 1.7 * x)
        dz = -0.12 * 2.3 * np.sin(3.1 * x + 0.6) * np.sin(2.3 * z) + 0.06 * 5.3 * np.cos(5.3 * z + 1.7 * x)
        pos = np.stack([x, y, z], -1)
        n = np.stack([-dx, np.ones_like(x), -dz], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        t = np.stack([np.ones_like(x), dx, np.zeros_like(x)], -1)
        t /= np.linalg.norm(t, axis=-1, keepdims=True)
        return pos, n, t
    return _merge([_grid(seg, seg, fn)])


def torus(seg=128):
    R, r = 0.75, 0.32

    def fn(u, v):
        a, b = u * 2 * math.pi, v * 2 * math.pi
        n = np.stack([np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)], -1)
        c = np.stack([np.cos(a) * R, np.zeros_like(a), np.sin(a) * R], -1)
        return c + n * r, n, np.stack([-np.sin(a), np.zeros_like(a), np.cos(a)], -1)
    p, n, t, uv, i = _grid(seg, seg // 2, fn)
    return _merge([(p, n, t, uv * [4, 1], i)])


def mat_ball(seg=128):
    """A sphere on a short pedestal, like a material ball."""
    s = sphere(seg)
    base = cylinder(seg // 2)
    bp = base[0] * [0.55, 0.12, 0.55] + [0, -1.05, 0]
    return _merge([s[:5], (bp, base[1], base[2], base[3], base[4])])


# The Scene menu: Standard primitives, then Extended primitives.
STANDARD = {"Cube": cube, "Cylinder": cylinder, "Hollow Box": hollow_box, "Inner Box": inner_box, "Plane": plane,
            "Plane (hi-res)": lambda: plane(512), "Sphere": sphere}
EXTENDED = {"Cloth": cloth, "Mat Ball": mat_ball, "Rounded Cube": lambda: cube(round_radius=0.25),
            "Rounded Cylinder": rounded_cylinder, "Sphere 2 tiles": lambda: sphere(tiles=2.0), "Torus": torus}
PRIMITIVES = {**STANDARD, **EXTENDED}
