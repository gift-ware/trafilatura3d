// Helpers shared by the effect and normal map filters (prepended to every shader of this folder).

// UV of the centre of output pixel c.
fn fx_uv(c: vec2i) -> vec2f { return (vec2f(c) + 0.5) / p.size_; }

// Output pixels per pixel of a 256 px image: distances in "pixels at 256" hold at every size.
fn fx_px() -> f32 { return p.size_.x / 256.0; }

fn fx_gray(v: f32) -> vec4f { return vec4f(v, v, v, 1.0); }

fn fx_rot(v: vec2f, a: f32) -> vec2f {
    let c = cos(a);
    let s = sin(a);
    return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Direction of an angle in turns, in UV (x right, y down), like the atomic Directional warp.
fn fx_dir(turns: f32) -> vec2f {
    let a = turns * 2.0 * PI;
    return vec2f(cos(a), sin(a));
}

// Normal maps: RGB in [0, 1] <-> vectors. Trafilatura 3D's convention is OpenGL (+Y up); DirectX flips green.
fn fx_decode_normal(t: vec4f, directx: bool) -> vec3f {
    var n = t.xyz * 2.0 - 1.0;
    if (directx) { n.y = -n.y; }
    return n;
}
fn fx_encode_normal(n: vec3f, directx: bool) -> vec3f {
    var m = n;
    if (directx) { m.y = -m.y; }
    return clamp(m * 0.5 + 0.5, vec3f(0.0), vec3f(1.0));
}

// Normal of a height field from its slopes: s = dh/dx, dh/dy in height units per UV, image rows grow
// downwards (so +Y, up, is -row).
fn fx_slope_normal(s: vec2f) -> vec3f { return normalize(vec3f(-s.x, s.y, 1.0)); }

// Reoriented normal mapping: detail `d` laid on base `b` (both unit vectors, Z up).
fn fx_rnm(b: vec3f, d: vec3f) -> vec3f {
    let t = b + vec3f(0.0, 0.0, 1.0);
    let u = d * vec3f(-1.0, -1.0, 1.0);
    return normalize(t * dot(t, u) / t.z - u);
}

fn fx_hash3(q: vec2i, seed: f32) -> vec3f {
    return vec3f(hash21(q, seed), hash21(q, seed + 17.17), hash21(q, seed + 39.71));
}

// Offset b - a between pixels, the short way round on tiled axes.
fn fx_wrap_delta(a: vec2f, b: vec2f) -> vec2f {
    var v = b - a;
    let d = p.size_;
    if (tile_h()) { v.x = v.x - d.x * round(v.x / d.x); }
    if (tile_v()) { v.y = v.y - d.y * round(v.y / d.y); }
    return v;
}

// Equirectangular (latitude-longitude) lookup coordinates of a direction, Z up, rotation in turns.
fn fx_latlong(d: vec3f, rotation: f32) -> vec2f {
    let x = select(d.x, 1e-9, d.x == 0.0 && d.y == 0.0);  // atan2(0, 0) is undefined (NaN on Metal) at the poles
    let u = fract(atan2(d.y, x) / (2.0 * PI) + 0.5 + rotation);
    let v = acos(clamp(d.z, -1.0, 1.0)) / PI;
    return vec2f(u, v);
}

// Orthonormal basis around a unit vector.
fn fx_basis(n: vec3f) -> mat3x3f {
    let a = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let t = normalize(cross(a, n));
    return mat3x3f(t, cross(n, t), n);
}

// Low-discrepancy 2D point i of n (Fibonacci lattice), rotated per pixel by r in [0, 1).
fn fx_fib(i: i32, n: i32, r: vec2f) -> vec2f {
    return fract(vec2f((f32(i) + 0.5) / f32(n), f32(i) * 0.618033988749) + r);
}

// Height slopes (dh/du, dh/dv per UV) from a 3x3 neighbourhood h[0..8] (rows top to bottom) spaced
// `e` UV apart: Sobel (sobel = true) or central differences.
fn fx_slopes3(h: array<f32, 9>, e: vec2f, sobel: bool) -> vec2f {
    if (sobel) {
        let dx = (h[2] + 2.0 * h[5] + h[8]) - (h[0] + 2.0 * h[3] + h[6]);
        let dy = (h[6] + 2.0 * h[7] + h[8]) - (h[0] + 2.0 * h[1] + h[2]);
        return vec2f(dx / (8.0 * e.x), dy / (8.0 * e.y));
    }
    return vec2f((h[5] - h[3]) / (2.0 * e.x), (h[7] - h[1]) / (2.0 * e.y));
}
