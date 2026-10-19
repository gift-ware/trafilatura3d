// What blending a layer over the stack needs: blend.wgsl (one layer) and the fused chains (nodes.BlendChain).

fn srgb_dec(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}
fn srgb_enc(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}
fn safe_div(a: vec3f, b: vec3f) -> vec3f {
    return a / select(b, vec3f(1e-5), abs(b) < vec3f(1e-5));
}
fn rgb2hsv(c: vec3f) -> vec3f {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    var h = 0.0;
    if (d > 1e-6) {
        if (mx == c.r) { h = (c.g - c.b) / d; } else if (mx == c.g) { h = 2.0 + (c.b - c.r) / d; } else { h = 4.0 + (c.r - c.g) / d; }
        h = fract(h / 6.0 + 1.0);
    }
    return vec3f(h, select(0.0, d / mx, mx > 1e-6), mx);
}
fn hsv2rgb(c: vec3f) -> vec3f {
    let k = vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let q = abs(fract(vec3f(c.x) + k) * 6.0 - 3.0);
    return c.z * mix(vec3f(1.0), clamp(q - 1.0, vec3f(0.0), vec3f(1.0)), c.y);
}
fn burn(b: vec3f, t: vec3f) -> vec3f { return 1.0 - safe_div(1.0 - b, t); }
fn dodge(b: vec3f, t: vec3f) -> vec3f { return safe_div(b, 1.0 - t); }
fn soft(b: vec3f, t: vec3f) -> vec3f {
    let d = select(sqrt(max(b, vec3f(0.0))), ((16.0 * b - 12.0) * b + 4.0) * b, b <= vec3f(0.25));
    return select(b + (2.0 * t - 1.0) * (d - b), b - (1.0 - 2.0 * t) * b * (1.0 - b), t <= vec3f(0.5));
}
fn overlay(b: vec3f, t: vec3f) -> vec3f {
    return select(1.0 - 2.0 * (1.0 - t) * (1.0 - b), 2.0 * t * b, b < vec3f(0.5));
}
// Normals in [-1, 1]. `unit`: a null vector (a filter's flat 0.5 on the Normal channel) is the flat normal.
fn unit(v: vec3f) -> vec3f { let l = length(v); return select(vec3f(0.0, 0.0, 1.0), v / l, l > 1e-6); }
fn whiteout(b: vec3f, t: vec3f) -> vec3f { return unit(vec3f(t.xy + b.xy, t.z * b.z)); }
fn rnm(base: vec3f, detail: vec3f) -> vec3f {
    let n1 = base + vec3f(0.0, 0.0, 1.0);
    let n2 = detail * vec3f(-1.0, -1.0, 1.0);
    return unit(n1 * dot(n1, n2) / max(n1.z, 1e-4) - n2);
}

fn blend(mode: i32, b: vec3f, t: vec3f) -> vec3f {
    switch mode {
        case 3: { return t; }                                   // Replace
        case 4: { return t * b; }                               // Multiply
        case 5: { return safe_div(b, t); }                      // Divide
        case 6: { return safe_div(t, b); }                      // Inverse Divide
        case 7: { return min(t, b); }                           // Darken (Min)
        case 8: { return max(t, b); }                           // Lighten (Max)
        case 9: { return b + t; }                               // Linear Dodge (Add)
        case 10: { return b - t; }                              // Subtract
        case 11: { return t - b; }                              // Inverse Subtract
        case 12: { return abs(b - t); }                         // Difference
        case 13: { return t + b - 2.0 * t * b; }                // Exclusion
        case 14: { return b + (t - 0.5) * 2.0; }                // Signed Addition (AddSub)
        case 15: { return overlay(b, t); }                      // Overlay
        case 16: { return 1.0 - (1.0 - t) * (1.0 - b); }        // Screen
        case 17: { return t + b - 1.0; }                        // Linear Burn
        case 18: { return burn(b, t); }                         // Color Burn
        case 19: { return dodge(b, t); }                        // Color Dodge
        case 20: { return soft(b, t); }                         // Soft Light
        case 21: { return overlay(t, b); }                      // Hard Light
        case 22: { return select(dodge(b, 2.0 * (t - 0.5)), burn(b, 2.0 * t), t < vec3f(0.5)); }  // Vivid Light
        case 23: { return b + 2.0 * t - 1.0; }                  // Linear Light
        case 24: { return select(max(b, 2.0 * (t - 0.5)), min(b, 2.0 * t), t < vec3f(0.5)); }     // Pin Light
        case 25: { let x = rgb2hsv(t); let y = rgb2hsv(b); return select(b, hsv2rgb(vec3f(x.x, y.y, y.z)), x.y > 1e-4); }  // Tint
        case 26: { let x = rgb2hsv(t); let y = rgb2hsv(b); return hsv2rgb(vec3f(y.x, x.y, y.z)); }  // Saturation
        case 27: { let x = rgb2hsv(t); let y = rgb2hsv(b); return hsv2rgb(vec3f(x.x, x.y, y.z)); }  // Color
        case 28: { let x = rgb2hsv(t); let y = rgb2hsv(b); return hsv2rgb(vec3f(y.x, y.y, x.z)); }  // Value
        default: { return t; }                                  // Normal
    }
}

// One layer over `below`: `top` its value and alpha, `m` its mask (blending.py's mode identifiers).
fn lay(below: vec4f, top: vec4f, m: f32, mode: i32, opacity: f32, linear: bool, clamp01: bool, normal: bool,
       keep: bool) -> vec4f {
    let replace = mode == 3;  // keep: a filter's result, its alpha ignored, the coverage below kept
    let k = select(top.a, 1.0, replace || keep) * opacity * m;
    var rgb: vec3f;
    if (normal) {  // the Normal channel: unit vectors, the result renormalized
        let nb = unit(below.rgb * 2.0 - 1.0);
        let nt = unit(top.rgb * 2.0 - 1.0);
        var r: vec3f;
        switch mode {
            case 29: { r = whiteout(nb, nt); }   // Normal Map Combine
            case 30: { r = rnm(nb, nt); }        // Normal Map Detail
            case 31: { r = rnm(nt, nb); }        // Normal Map Inverse Detail
            case 0, 3: { r = nt; }
            default: { r = unit(blend(mode, below.rgb, top.rgb) * 2.0 - 1.0); }
        }
        rgb = unit(mix(nb, r, k)) * 0.5 + 0.5;
    } else {
        var b = below.rgb;
        var t = top.rgb;
        if (linear) { b = srgb_dec(b); t = srgb_dec(t); }
        var r = blend(mode, b, t);
        if (clamp01) { r = clamp(r, vec3f(0.0), vec3f(1.0)); }
        rgb = mix(b, r, k);
        if (linear) { rgb = srgb_enc(rgb); }
    }
    var a = select(below.a + (1.0 - below.a) * k, mix(below.a, top.a, opacity * m), replace);
    if (keep) { a = below.a; }
    return vec4f(rgb, clamp(a, 0.0, 1.0));
}
