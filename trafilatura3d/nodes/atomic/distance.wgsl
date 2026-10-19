fn delta(a: vec2f, b: vec2f) -> vec2f {
    var v = abs(a - b);
    let d = vec2f(size());
    if (tile_h()) { v.x = min(v.x, d.x - v.x); }
    if (tile_v()) { v.y = min(v.y, d.y - v.y); }
    return v;
}

fn metric(v: vec2f) -> f32 {
    if (p.distance_mode == 1) { return v.x + v.y; }
    if (p.distance_mode == 2) { return max(v.x, v.y); }
    return length(v);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = seed_at(c);
    if (s.x < 0.0) { store(c, vec4f(0.0, 0.0, 0.0, 1.0)); return; }  // no white pixel at all
    let dist = metric(delta(s.xy, vec2f(c))) * 256.0 / f32(d.x);     // in pixels of a 256 px image
    let fade = clamp(1.0 - dist / max(p.maximum_distance, 1e-4), 0.0, 1.0);
    var o = vec4f(fade, fade, fade, 1.0);
    if (SOURCE_INPUT_CONNECTED) {
        // Take the source where the closest white texel is, faded or solid.
        let src = texel_source_input(vec2i(floor((s.xy + 0.5) * vec2f(size_source_input()) / vec2f(d))));
        let inside = select(0.0, 1.0, dist <= p.maximum_distance);
        o = select(src * fade, src * inside, p.combine_source_distance == 1);
        if (SOURCE_INPUT_GRAY) { o = vec4f(o.r, o.r, o.r, 1.0); }
    }
    if (OUT_OUTPUT_GRAY) { o = vec4f(o.r, o.r, o.r, 1.0); }
    store(c, o);
}
