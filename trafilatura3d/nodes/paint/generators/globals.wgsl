// Stage 2 of the staged mask generators (_common.Staged): the combined mask, already blurred by Global Blur,
// then Global Balance, Global Contrast and Global Invert.
fn global_final(c: vec2i) -> vec4f {
    var m = sample_mix_(uv_of(c)).r;
    m = g_contrast(g_balance(m, p.global_balance), p.global_contrast);
    m = g_invert(m, p.global_invert);
    return vec4f(vec3f(m), 1.0);
}

// A layer's own adjustments after its blur: Balance, Contrast, Brightness.
fn g_adjust(x: f32, balance: f32, contrast: f32, brightness: f32) -> f32 {
    return clamp(g_contrast(g_balance(x, balance), contrast) + brightness, 0.0, 1.0);
}

// Curvature Mode: 0 Edges (convex half stretched to 0-1), 1 Cavities (concave half), 2 Dual (both), 3 Unprocessed.
fn g_curvature_mode(x: f32, mode: i32) -> f32 {
    switch mode {
        case 0: { return clamp((x - 0.5) * 2.0, 0.0, 1.0); }
        case 1: { return clamp((0.5 - x) * 2.0, 0.0, 1.0); }
        case 2: { return clamp(abs(x - 0.5) * 2.0, 0.0, 1.0); }
        default: { return x; }
    }
}
