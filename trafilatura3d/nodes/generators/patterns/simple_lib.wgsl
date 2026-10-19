// Helpers of the simple tiled patterns (bricks, checker, mesh, weave, fibers, gaussians).

// Grid coordinates: `n` cells per uv unit; rotated by 45 degrees the grid axes are the diagonals
// (u + v, v - u) * n, which still tiles for an integer n (cells sqrt(2) smaller).
fn grid(uv: vec2f, n: f32, rot45: i32) -> vec2f {
    if (rot45 == 1) { return vec2f(uv.x + uv.y, uv.y - uv.x) * n; }
    return uv * n;
}
// One output pixel in grid units.
fn grid_px(n: f32, rot45: i32) -> f32 { return pixel_uv() * n * select(1.0, 1.41421356, rot45 == 1); }
// Integer cell count along each frame axis (for wrapping random cells), n cells per uv unit.
fn grid_cells(n: f32, nse: i32) -> vec2i { return max(vec2i(round(n * frame_size(nse))), vec2i(1)); }
fn imod2(x: f32) -> f32 { return x - 2.0 * floor(x * 0.5); }
// Circular (pillow) profile of a 0-1 ramp.
fn round_profile(t: f32) -> f32 { let a = clamp(t, 0.0, 1.0); return sqrt(a * (2.0 - a)); }

// Running-bond bricks: bx bricks per row, by rows per tiling unit, odd rows shifted by half a brick.
// gap = joint half width, bev = bevel width (both as fractions of the brick height), pillow =
// circular profile on both axes instead of a straight bevel.
fn brick_simple(uv: vec2f, n: f32, bx: f32, by: f32, gap: f32, bev: f32, pillow: bool) -> f32 {
    let cells = n * vec2f(bx, by);
    let g = uv * cells;
    let row = floor(g.y);
    let x = g.x + 0.5 * imod2(row);
    let f = fract(vec2f(x, g.y));
    let h = 1.0 / cells.y;                              // brick height in uv
    let dd = min(f, 1.0 - f) / cells - gap * h;         // distance inside the joint, uv
    let px = pixel_uv();
    let d = min(dd.x, dd.y);
    if (pillow) {
        let b = max(bev * h, px);
        return edge(d, px) * round_profile(dd.x / b) * round_profile(dd.y / b);
    }
    let r = clamp((d + 0.5 * px) / max(bev * h, px), 0.0, 1.0);
    return r * r * (3.0 - 2.0 * r);
}

// Sum of round Gaussian blobs (standard deviation `sigma`) on staggered rows, in grid units: row r at
// y = origin.y + r * pitch.y, its blobs at x = origin.x + (m + 0.5 * (r mod 2)) * pitch.x.
fn gauss_rows(g: vec2f, pitch: vec2f, origin: vec2f, sigma: f32) -> f32 {
    let h = (g - origin) / pitch;
    let r0 = round(h.y);
    var v = 0.0;
    for (var dr = -2; dr <= 2; dr++) {
        let r = r0 + f32(dr);
        let sh = 0.5 * imod2(r);
        let m0 = round(h.x - sh);
        for (var dm = -2; dm <= 2; dm++) {
            let d = g - origin - vec2f(m0 + f32(dm) + sh, r) * pitch;
            v += exp(-dot(d, d) / (2.0 * sigma * sigma));
        }
    }
    return v;
}

// Capsule along x: a segment of half length `hl` swept by an elliptic cap (radius rad.x along, rad.y
// across), height 1 - r^pw over the normalized distance r to the segment (0 outside).
fn capsule(d: vec2f, hl: f32, rad: vec2f, pw: f32) -> f32 {
    let r = length(vec2f(max(abs(d.x) - hl, 0.0) / rad.x, d.y / rad.y));
    return max(1.0 - pow(r, pw), 0.0);
}

// Mesh 1 / Mesh 2: a blob on every grid point, horizontal where i + j is even, vertical elsewhere,
// the highest one winning. Fat: a parabolic capsule; thin: a narrow spindle, exp(-|x|^2.16 - |y|^1.56)
// over its two radii.
fn mesh_cells(g: vec2f, fat: bool) -> f32 {
    let i0 = round(g);
    var v = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let k = i0 + vec2f(f32(i), f32(j));
            var o = g - k;
            if (imod2(k.x + k.y) == 1.0) { o = o.yx; }
            if (fat) {
                v = max(v, capsule(o, 21.0 / 64.0, vec2f(30.25 / 64.0), 2.0));
            } else {
                let a = abs(o) / vec2f(0.49, 0.152);
                v = max(v, 0.96 * exp(-pow(a.x, 2.16) - pow(a.y, 1.56)));
            }
        }
    }
    return v;
}

// Fibers 1 / Fibers 2: rope strands, tilted elliptic blobs (half axes `axes`, the long one turned by
// `th` radians from x, clockwise on screen) on the lattice origin + pitch * (i, j), the highest one
// winning. Profile: Gaussian exp(-q^2), else (1 - q^2)^2, q = the elliptic distance.
fn strands(g: vec2f, pitch: vec2f, origin: vec2f, axes: vec2f, th: f32, gaussian: bool) -> f32 {
    let i0 = round((g - origin) / pitch);
    let cs = vec2f(cos(th), sin(th));
    var v = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -3; i <= 3; i++) {
            let o = g - origin - (i0 + vec2f(f32(i), f32(j))) * pitch;
            let e = vec2f(cs.x * o.x + cs.y * o.y, cs.x * o.y - cs.y * o.x) / axes;
            let q2 = dot(e, e);
            let s = max(1.0 - q2, 0.0);
            v = max(v, select(s * s, exp(-q2), gaussian));
        }
    }
    return v;
}
