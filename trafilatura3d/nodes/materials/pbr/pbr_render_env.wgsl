@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // Average the footprint of a texel of the small panorama (4 x 4 taps).
    let uv = uv_of(c, d);
    var acc = vec3f(0.0);
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let o = (vec2f(f32(i), f32(j)) - 1.5) / 4.0 / vec2f(d);
            acc += sample_environment_map(vec2f(uv.x + o.x, clamp(uv.y + o.y, 0.0, 1.0))).rgb;
        }
    }
    store(c, vec4f(acc / 16.0, 1.0));
}
