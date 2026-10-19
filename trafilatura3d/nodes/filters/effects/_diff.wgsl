// Diffusion value encoding. Color / grayscale: the value itself (normal maps renormalized). UV: on a
// tiled axis the coordinate as an angle (cos, sin), so averages wrap around 0/1 (0.1 and 0.8 average
// to 0.95); else (u, 0).
fn diff_encode(t: vec4f) -> vec4f {
    if (p.mode != 2) { return t; }
    let a = t.xy * 2.0 * PI;
    var e = vec4f(t.x, 0.0, t.y, 0.0);
    if (tile_h()) { e.x = cos(a.x); e.y = sin(a.x); }
    if (tile_v()) { e.z = cos(a.y); e.w = sin(a.y); }
    return e;
}

fn diff_decode(e: vec4f) -> vec4f {
    if (p.mode != 2) { return e; }
    var u = e.x;
    var v = e.z;
    if (tile_h()) { u = fract(atan2(e.y, e.x) / (2.0 * PI) + 1.0); }
    if (tile_v()) { v = fract(atan2(e.w, e.z) / (2.0 * PI) + 1.0); }
    return vec4f(u, v, 0.0, 1.0);
}

fn diff_known(c: vec2i) -> bool {
    if (p.alpha_mask == 1) { return load_source(c).a >= 0.5; }
    return load_mask(c).r >= 0.5;
}
