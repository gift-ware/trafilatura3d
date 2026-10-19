// Signed distance to a shape's edge from the jump flooding seeds of filters/effects/_fx.nearest_seeds (xy = nearest
// pixel inside, zw = nearest outside, -1 = none), in output pixels, negative inside; wraps on tiled axes.
fn wrap_delta(a: vec2f, b: vec2f) -> vec2f {
    var v = b - a;
    let d = p.size_;
    if (tile_h()) { v.x = v.x - d.x * round(v.x / d.x); }
    if (tile_v()) { v.y = v.y - d.y * round(v.y / d.y); }
    return v;
}

fn edge_sd(c: vec2i) -> f32 {
    let s = texel_seeds(c);
    let me = vec2f(c);
    if (s.z < 0.0) { return -1e6; }  // no outside pixel at all
    if (s.x < 0.0) { return 1e6; }   // no inside pixel at all
    if (all(s.xy == me)) { return -(length(wrap_delta(me, s.zw)) - 0.5); }
    return length(wrap_delta(me, s.xy)) - 0.5;
}
