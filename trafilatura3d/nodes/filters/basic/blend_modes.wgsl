// The library's blend filters, by the modes of Blend (Deprecated) (mode values: 1 Copy ... 19
// Luminosity); `mode_()` is the node's (one mode, or its Blending mode).
fn blend_formula(b: vec3f, f: vec3f, mode: i32) -> vec3f {
    // Copy, Add, Subtract, Multiply, Add sub, Max, Min: the atomic Blend's 0-6; Overlay, Screen, Soft light its 9-11.
    if (mode <= 7) { return blend_atomic(b, f, mode - 1); }
    switch mode {
        case 9, 10: { return blend_atomic(b, f, mode); }
        case 14: { return blend_atomic(b, f, 11); }
        case 11: { return max(f + b - 1.0, vec3f(0.0)); }                                       // Linear burn
        case 12: { return select(clamp(1.0 - (1.0 - b) / max(f, vec3f(1e-6)), vec3f(0.0), vec3f(1.0)),
                                 select(vec3f(0.0), vec3f(1.0), b >= vec3f(1.0)), f <= vec3f(0.0)); }  // Color burn
        case 13: { return select(clamp(b / max(1.0 - f, vec3f(1e-6)), vec3f(0.0), vec3f(1.0)),
                                 select(vec3f(1.0), vec3f(0.0), b <= vec3f(0.0)), f >= vec3f(1.0)); }  // Color dodge
        case 15: { return select(1.0 - 2.0 * (1.0 - b) * (1.0 - f), 2.0 * b * f, f < vec3f(0.5)); }  // Hard light
        case 16: { return set_lum(f, lum_w3c(b)); }                                             // Color
        case 17: { return abs(f - b); }                                                         // Difference
        case 18: {  // Exclusion: the clamped sum less a share of the product, as the upper of two lines
            // in (sum, product); slopes fitted on the output (exact over the whole 0-1 square).
            let s = min(b + f, vec3f(1.0));
            let m = b * f;
            return max(s - 1.25 * m, 0.5 * (s - m));
        }
        case 19: { return set_lum(b, lum_w3c(f)); }                                             // Luminosity
        default: { return f; }                                                                  // Dissolve
    }
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let b = load_background(c);
    let f = load_foreground(c);
    var k = clamp(p.opacity * load_mask(c).r, 0.0, 1.0);
    if (mode_() == 8) { k = select(0.0, 1.0, hash21(c, p.seed_ + 17.0) < k); }
    let rgb = blend_formula(b.rgb, f.rgb, mode_());
    var o: vec4f;
    if (p.alpha_blending != 0) {
        let fa = select(f.a, 1.0, FOREGROUND_GRAY) * k;
        o = vec4f(mix(b.rgb, rgb, fa), fa + b.a * (1.0 - fa));
    } else {
        o = vec4f(mix(b.rgb, rgb, k), b.a);
    }
    if (OUT_OUTPUT_GRAY) { o = gray4(o.r); }
    store(c, o);
}
