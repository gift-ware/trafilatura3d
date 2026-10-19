// b (1 - t) + x t: exactly b at t = 0 and x at t = 1 (mix() computes b + t (x - b), which can miss x by an ulp: a
// 1.0000001 that a later Auto Levels and Histogram Scan turn into a visible difference).
fn lerp4(b: vec4f, x: vec4f, t: f32) -> vec4f { return b * (1.0 - t) + x * t; }
fn lerp3(b: vec3f, x: vec3f, t: f32) -> vec3f { return b * (1.0 - t) + x * t; }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let b = load_background(c);
    let f = load_foreground(c);
    var k = p.opacity * load_opacity(c).r;
    let crop = p.cropping_area;
    if (uv.x < crop.x || uv.x > crop.y || uv.y < crop.z || uv.y > crop.w) { k = 0.0; }
    // Measured on the original: the alpha is the foreground's over the background's in every alpha blending; Switch
    // mixes all four channels.
    if (p.blending_mode == 7) {                                               // Switch
        store(c, lerp4(b, f, k));
        return;
    }
    var x = blend_atomic(b.rgb, f.rgb, p.blending_mode);
    let s = select(f.a, 1.0, FOREGROUND_GRAY) * k;
    let a = s + b.a * (1.0 - s);
    var rgb: vec3f;
    switch p.alpha_blending {
        case 1: { rgb = lerp3(b.rgb, x, k); }                                   // Ignore alpha
        case 2: {                                                             // Straight alpha (W3C compositing)
            if (p.q_output > 0.0) { x = clamp(x, vec3f(0.0), vec3f(1.0)); }  // an integer format clamps the blend first
            rgb = (s * lerp3(f.rgb, x, b.a) + b.a * b.rgb * (1.0 - s)) / max(a, 1e-5);
        }
        case 3: {                                                             // Premultiplied: Copy takes it so
            rgb = select(lerp3(b.rgb, x, s), f.rgb * k + b.rgb * (1.0 - s), p.blending_mode == 0);
        }
        default: { rgb = lerp3(b.rgb, x, s); }                                  // Use source alpha
    }
    store(c, vec4f(rgb, a));
}
