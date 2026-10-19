// ---- weathering/_weather.wgsl ----
fn wsn_dec(c: vec3f) -> vec3f { return normalize(c * 2.0 - 1.0 + vec3f(0.0, 1e-5, 0.0)); }
fn convex_of(curv: f32) -> f32 { return sat((curv - 0.5) * 2.0); }
fn concave_of(curv: f32) -> f32 { return sat((0.5 - curv) * 2.0); }
// Crack network: 1 on the cracks of warped cells (`scale` cells), `warp` bends them.
fn crack_pattern(uv: vec2f, scale: f32, warp: f32, seed: f32) -> f32 {
    let n = i32(max(scale, 1.0));
    let w = vec2f(fbm(uv, n, 3, 0.5, seed + 1.0), fbm(uv, n, 3, 0.5, seed + 2.0)) - 0.5;
    let cl = cells(fract(uv + w * warp * 0.6 / f32(n) * 4.0), n, 1.0, seed);
    return smoothstep(0.08, 0.0, cl.w);
}
// Scratches on sharp edges: `scale` sets their density, `warp` bends them.
fn edge_scratches(uv: vec2f, scale: f32, warp: f32, seed: f32) -> f32 {
    let w = (vnoise(uv, vec2i(8), seed + 3.0) - 0.5) * warp;
    return scratches(fract(uv + vec2f(w, -w) * 0.1), i32(max(scale, 1.0)) * 2, 1.2, 0.5 + w * 3.0, 1.2, 0.05, seed);
}
// Normal of a relief given its values at the pixel and one pixel right / up (image y down).
fn relief_normal(c0: f32, cx: f32, cy: f32, k: f32) -> vec3f { return n_slopes(cx - c0, cy - c0, k); }
