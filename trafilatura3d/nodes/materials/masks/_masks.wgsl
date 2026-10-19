// ---- masks/_masks.wgsl ----
fn convex(curv: f32) -> f32 { return sat((curv - 0.5) * 2.0); }
// A level control as a window on x in 0..1: level 0.5 keeps x, above it the window [max(0, 1 - 2 level),
// min(1, 2 - 2 level)] slides down (more passes), below it up (less); 0 and 1 are steps. Not clamped.
fn level_window(x: f32, level: f32) -> f32 {
    let lo = max(0.0, 1.0 - 2.0 * level);
    let hi = min(1.0, 2.0 - 2.0 * level);
    return (x - lo) / max(hi - lo, 1e-4);
}
// Edge wear's occlusion factor: the AO through a saturating curve (tanh, fitted on the output), then the level
// window of 1 - AO Masking (0: none, 1: all).
fn occlusion(ao: f32, masking: f32) -> f32 {
    let k = 1.8;
    return sat(level_window(tanh(k * sat(ao)) / tanh(k), 1.0 - masking));
}
// Edge wear's last step: the Wear Level window, then Wear Contrast about 0.5.
fn wear(x: f32, level: f32, contrast: f32) -> f32 {
    let w = sat(level_window(x, level));
    return sat((w - 0.5) / max(1.0 - contrast, 0.002) + 0.5);
}
fn concave(curv: f32) -> f32 { return sat((0.5 - curv) * 2.0); }
fn wsn(c: vec3f) -> vec3f { return normalize(c * 2.0 - 1.0 + vec3f(0.0, 1e-5, 0.0)); }
// Upward direction in UV space from a height (position) field sampled around uv: the way a drip
// runs is the opposite. Flat fields: towards the top of the image.
fn up_dir(hc: f32, hx: f32, hy: f32) -> vec2f {
    let g = vec2f(hx - hc, hy - hc);
    if (length(g) < 1e-5) { return vec2f(0.0, -1.0); }
    return normalize(g);
}
// Noise of a kind (Spots, Cloud, Moisture, White Noise) at a scale index 0..10.
fn noise_kind(kind: i32, uv: vec2f, scale: i32, seed: f32) -> f32 {
    let n = i32(exp2(f32(clamp(scale, 0, 10)) * 0.6 + 1.0));
    switch kind {
        case 0: { let s = cells(uv, n, 1.0, seed); return smoothstep(0.6, 0.1, s.x) * (0.5 + 0.5 * s.z); }
        case 2: {
            let w = vec2f(fbm(uv, n, 3, 0.5, seed + 4.0), fbm(uv, n, 3, 0.5, seed + 8.0)) - 0.5;
            return smoothstep(0.35, 0.65, fbm(fract(uv + w * 0.3), n, 5, 0.6, seed));
        }
        case 3: { return hash21(vec2i(uv * f32(n) * 16.0), seed); }
        default: { return fbm(uv, n, 5, 0.55, seed); }
    }
}
