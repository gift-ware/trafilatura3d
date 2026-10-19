// Shape effects (stroke, glow, drop shadow) on an `input`: a grayscale mask, or a color image with
// alpha. Distances in output pixels.

// How far a falloff of Width w reaches (128 w^2 pixels at 256), in output pixels.
fn shape_reach(w: f32) -> f32 { return 0.5 * w * w * p.size_.x; }

// The shape as straight (not premultiplied) color and alpha.
fn shape_layer(c: vec2i, mask_color: vec4f, premultiplied: bool) -> vec4f {
    let t = load_input(c);
    if (INPUT_GRAY) { return vec4f(mask_color.rgb, clamp(t.r, 0.0, 1.0) * mask_color.a); }
    var rgb = t.rgb;
    if (premultiplied && t.a > 1e-5) { rgb = rgb / t.a; }
    return vec4f(rgb, clamp(t.a, 0.0, 1.0));
}

// Porter-Duff "over" of straight-alpha layers.
fn over(top: vec4f, bottom: vec4f) -> vec4f {
    let a = top.a + bottom.a * (1.0 - top.a);
    if (a < 1e-6) { return vec4f(0.0); }
    return vec4f((top.rgb * top.a + bottom.rgb * bottom.a * (1.0 - top.a)) / a, a);
}

fn finish(v: vec4f, premultiply: bool) -> vec4f {
    if (premultiply) { return vec4f(v.rgb * v.a, v.a); }
    return v;
}
