"""The Warp projection's surface (Warp projection page: a grid of points deforming the projected image). The grid's
points are in the projection's space; the surface goes through them, smooth (Catmull-Rom in both directions:
Automatic tangents) or straight between them (bilinear). It is sampled on a fine grid that the fill shader reads
to find, for every texel, the image coordinates of the nearest point of the surface (fill.wgsl).

Ours where the page is silent: the interpolation (it names a Bezier patch with tangents; the edition of tangents
is not offered), the default grid (3 x 3 cells over the projection's square), the sampling.
"""

from __future__ import annotations

import numpy as np

from trafilatura3d.paint.model import warp_grid

SAMPLES = 65  # samples per side of the surface sampled for the shader


def _catmull_rom(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points at parameters `t` (0 .. len(ctrl) - 1) of the Catmull-Rom spline through `ctrl` (k, ..., 3), ends
    clamped."""
    k = ctrl.shape[0]
    i = np.clip(np.floor(t).astype(int), 0, k - 2)
    f = (t - i)[(...,) + (None,) * (ctrl.ndim - 1)]
    p0, p1, p2, p3 = (ctrl[np.clip(i + d, 0, k - 1)] for d in (-1, 0, 1, 2))
    return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)


def _linear(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    k = ctrl.shape[0]
    i = np.clip(np.floor(t).astype(int), 0, k - 2)
    f = (t - i)[(...,) + (None,) * (ctrl.ndim - 1)]
    return ctrl[i] * (1 - f) + ctrl[i + 1] * f


def points(pr) -> np.ndarray:
    """The grid's points (rows + 1, cols + 1, 3) of a fill's projection (the flat grid when not edited)."""
    pts = pr.warp_points or warp_grid(pr.warp_rows, pr.warp_cols)
    return np.asarray(pts, np.float64).reshape(pr.warp_rows + 1, pr.warp_cols + 1, 3)


def surface(pr, n: int = SAMPLES) -> np.ndarray:
    """The surface sampled on n x n parameters (rows: v from the top, columns: u), (n, n, 3) float32."""
    grid = points(pr)
    rows, cols = grid.shape[0] - 1, grid.shape[1] - 1
    interp = _catmull_rom if pr.warp_tangents else _linear
    tu = np.linspace(0, cols, n)
    tv = np.linspace(0, rows, n)
    along_u = interp(np.swapaxes(grid, 0, 1), tu)  # (n, rows + 1, 3): every row's curve sampled along u
    s = interp(np.swapaxes(along_u, 0, 1), tv)  # (n v, n u, 3)
    return s.astype(np.float32)
