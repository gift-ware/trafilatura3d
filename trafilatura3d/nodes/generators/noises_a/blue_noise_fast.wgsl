fn white(c: vec2i) -> f32 {
    let q = vec2f(wrap(c, size())) + 0.5;
    let r = rot2(q, (1.0 - clamp(p.rotation, 0.0, 1.0)) * 0.125);
    return hash21(vec2i(floor(r)), p.seed_);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var mean = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            mean += white(c + vec2i(i, j)) * select(1.0, 2.0, i == 0 && j == 0);
        }
    }
    let hp = white(c) - mean / 10.0;
    let v = clamp(0.5 + hp * 1.6, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
