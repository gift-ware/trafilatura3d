"""A BVH over triangles for GPU ray tracing (the path tracer, the bakers): triangles sorted by the
Morton code of their centroid, an implicit balanced binary tree (heap layout: node k has children 2k+1
and 2k+2, leaves of LEAF triangles), bounds built level by level in numpy — milliseconds, no Python loop
per node, lower quality than SAH."""

from __future__ import annotations

import math

import numpy as np

LEAF = 4

# WGSL traversal. The including shader binds `verts` (3 vec4 per vertex, position first), `tris` (vec4u:
# 3 indices + a user value) and `nodes` (2 vec4 per node: min, max), and defines `bvh_leaf_base` and
# `bvh_tri_count`. `bvh_cull` (default 0) ignores backface hits: 1 all, 2 those of triangles whose user value
# has bit 0 set (the bakers' Ignore Backface "By Mesh Name").
WGSL = """
struct BvhHit { t: f32, tri: u32, b: vec2f };
var<private> bvh_cull: u32;
fn bvh_box(o: vec3f, inv: vec3f, lo: vec3f, hi: vec3f, tmax: f32) -> f32 {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return select(1e30, max(tn, 0.0), tn <= tf && tf > 0.0 && tn < tmax);
}
fn bvh_trace(o: vec3f, d: vec3f, tmin: f32, tmax: f32, any_hit: bool) -> BvhHit {
    var best = BvhHit(tmax, 0xffffffffu, vec2f(0.0));
    let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
    var stack: array<u32, 40>;
    stack[0] = 0u;
    var sp = 1u;
    loop {
        if (sp == 0u) { break; }
        sp -= 1u;
        let n = stack[sp];
        if (bvh_box(o, inv, nodes[n * 2u].xyz, nodes[n * 2u + 1u].xyz, best.t) >= best.t) { continue; }
        if (n >= bvh_leaf_base) {
            let first = (n - bvh_leaf_base) * 4u;
            for (var k = 0u; k < 4u; k++) {
                let ti = first + k;
                if (ti >= bvh_tri_count) { break; }
                let t = tris[ti];
                let a = verts[t.x * 3u].xyz;
                let e1 = verts[t.y * 3u].xyz - a;
                let e2 = verts[t.z * 3u].xyz - a;
                let q = cross(d, e2);
                let det = dot(e1, q);
                if (abs(det) < 1e-14) { continue; }
                if (det < 0.0 && (bvh_cull == 1u || (bvh_cull == 2u && (t.w & 1u) == 1u))) { continue; }  // backface
                let f = 1.0 / det;
                let s = o - a;
                let bu = dot(s, q) * f;
                if (bu < 0.0 || bu > 1.0) { continue; }
                let r = cross(s, e1);
                let bv = dot(d, r) * f;
                if (bv < 0.0 || bu + bv > 1.0) { continue; }
                let tt = dot(e2, r) * f;
                if (tt <= tmin || tt >= best.t) { continue; }
                best = BvhHit(tt, ti, vec2f(bu, bv));
                if (any_hit) { return best; }
            }
        } else if (sp + 2u <= 40u) {
            stack[sp] = n * 2u + 2u;
            stack[sp + 1u] = n * 2u + 1u;
            sp += 2u;
        }
    }
    return best;
}
"""


def build(tri_pos: np.ndarray):
    """(order, node bounds, leaf base index) for triangles (n, 3, 3): reorder the triangles with
    `order` before uploading them."""
    order, leaves = morton_bvh(tri_pos.mean(1))
    return order, bvh_bounds(tri_pos[order], leaves), leaves - 1


def morton_bvh(centroids: np.ndarray):
    """Triangle order and an implicit balanced BVH (heap layout) over Morton-sorted triangles:
    returns (order, leaf_count_pow2). Bounds are filled by `bvh_bounds`."""
    lo, hi = centroids.min(0), centroids.max(0)
    q = ((centroids - lo) / np.maximum(hi - lo, 1e-9) * 1023).astype(np.uint64)

    def spread(x):
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x
    code = spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)
    order = np.argsort(code, kind="stable")
    leaves = (len(centroids) + LEAF - 1) // LEAF
    return order, 1 << max(0, math.ceil(math.log2(max(leaves, 1))))


def bvh_bounds(tri_pos: np.ndarray, leaves_pow2: int) -> np.ndarray:
    """Node bounds (2 * nodes, 4) float32 for triangles already in BVH order: (n, 3, 3)."""
    n = len(tri_pos)
    tmin, tmax = tri_pos.min(1), tri_pos.max(1)
    pad = leaves_pow2 * LEAF - n
    tmin = np.concatenate([tmin, np.full((pad, 3), np.inf, np.float32)])
    tmax = np.concatenate([tmax, np.full((pad, 3), -np.inf, np.float32)])
    level_min = tmin.reshape(leaves_pow2, LEAF, 3).min(1)
    level_max = tmax.reshape(leaves_pow2, LEAF, 3).max(1)
    total = 2 * leaves_pow2 - 1
    bmin = np.empty((total, 3), np.float32)
    bmax = np.empty((total, 3), np.float32)
    start, count = leaves_pow2 - 1, leaves_pow2
    bmin[start:start + count], bmax[start:start + count] = level_min, level_max
    while count > 1:
        level_min = np.minimum(level_min[0::2], level_min[1::2])
        level_max = np.maximum(level_max[0::2], level_max[1::2])
        count //= 2
        start = count - 1
        bmin[start:start + count], bmax[start:start + count] = level_min, level_max
    empty = ~np.isfinite(bmin).all(1)
    # Never hit: a point far away. (An inverted box, min 1e30 / max -1e30, passes the slab test as an
    # infinite box: every ray walked the empty half of the tree.)
    bmin[empty], bmax[empty] = 1e30, 1e30
    out = np.zeros((total * 2, 4), np.float32)
    out[0::2, :3], out[1::2, :3] = bmin, bmax
    return out
