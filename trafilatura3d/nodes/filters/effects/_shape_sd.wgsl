// Signed distance to the shape's edge from jump flooding seeds: negative inside.
fn shape_sd(c: vec2i) -> f32 {
    let s = texel_seeds(c);
    let me = vec2f(c);
    if (s.z < 0.0) { return -1e6; }  // no outside pixel at all
    if (s.x < 0.0) { return 1e6; }   // no inside pixel at all
    let inside = all(s.xy == me);
    if (inside) { return -(length(fx_wrap_delta(me, s.zw)) - 0.5); }
    return length(fx_wrap_delta(me, s.xy)) - 0.5;
}
