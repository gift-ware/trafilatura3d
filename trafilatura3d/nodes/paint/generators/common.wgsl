// Helpers of the generators of this folder (no parameters, no inputs: everything passed as arguments).

// Balance: moves the midpoint, 0.5 maps to `b` (0.5 = unchanged, towards 1 brighter, towards 0 darker).
fn g_balance(x: f32, b: f32) -> f32 {
    let bb = clamp(b, 0.001, 0.999);
    return pow(clamp(x, 0.0, 1.0), log(bb) / log(0.5));
}
// Contrast around 0.5: 0 unchanged, 1 nearly a threshold, -1 flat gray.
fn g_contrast(x: f32, k: f32) -> f32 {
    let s = select(1.0 + k, 1.0 + k * 8.0, k >= 0.0);
    return clamp((x - 0.5) * s + 0.5, 0.0, 1.0);
}
fn g_invert(x: f32, on: i32) -> f32 { return select(x, 1.0 - x, on != 0); }

// The grayscale conversion methods (the Grayscale Conversion generator's list, GRAY_METHODS).
fn g_gray(rgb: vec3f, method: i32) -> f32 {
    let hi = max(rgb.r, max(rgb.g, rgb.b));
    let lo = min(rgb.r, min(rgb.g, rgb.b));
    switch method {
        case 0: { return (hi + lo) * 0.5; }                        // Desaturation
        case 2: { return (rgb.r + rgb.g + rgb.b) / 3.0; }          // Average
        case 3: { return hi; }                                      // Max
        case 4: { return lo; }                                      // Min
        case 5: { return rgb.r; }
        case 6: { return rgb.g; }
        case 7: { return rgb.b; }
        default: { return dot(rgb, vec3f(0.2126, 0.7152, 0.0722)); }  // Luma
    }
}

// Layer blending of the mask generators (BLEND_MODES, the Blend names): `f` over `b` at `opacity`.
fn g_blend(b: f32, f: f32, mode: i32, opacity: f32) -> f32 {
    var r = f;
    switch mode {
        case 1: { r = b + f; }                                                  // Add
        case 2: { r = b - f; }                                                  // Subtract
        case 3: { r = b * f; }                                                  // Multiply
        case 4: { r = b + 2.0 * f - 1.0; }                                      // Add Sub
        case 5: { r = max(b, f); }                                              // Max
        case 6: { r = min(b, f); }                                              // Min
        case 7: { r = b / max(f, 1e-4); }                                       // Divide
        case 8: { r = select(1.0 - 2.0 * (1.0 - b) * (1.0 - f), 2.0 * b * f, b < 0.5); }  // Overlay
        case 9: { r = 1.0 - (1.0 - b) * (1.0 - f); }                            // Screen
        case 10: { r = (1.0 - 2.0 * f) * b * b + 2.0 * f * b; }                 // Soft Light
        default: { r = f; }                                                     // Copy
    }
    return clamp(mix(b, r, clamp(opacity, 0.0, 1.0)), 0.0, 1.0);
}

// Up to three directional gradients (each 0-1) blended in order onto black, each at its amount.
fn g_three(v: vec3f, amount: vec3f, inv: vec3i, mode: vec3i) -> f32 {
    var acc = 0.0;
    for (var k = 0; k < 3; k++) {
        acc = g_blend(acc, g_invert(v[k], inv[k]), mode[k], amount[k]);
    }
    return acc;
}

// Normal maps (world space or tangent space) from [0, 1] to vectors.
fn g_normal(t: vec4f) -> vec3f { return t.xyz * 2.0 - 1.0; }

fn g_rot(v: vec2f, turns: f32) -> vec2f {
    let a = turns * 2.0 * PI;
    let c = cos(a);
    let s = sin(a);
    return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Ridged value noise in 3D (fibers, streaks): 1 on the ridges.
fn g_ridged(x: vec3f, seed: f32) -> f32 { return 1.0 - abs(vnoise(x, seed) * 2.0 - 1.0); }
