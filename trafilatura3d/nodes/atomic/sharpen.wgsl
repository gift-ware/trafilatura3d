@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let e = 1.0 / vec2f(d);
    let v = sample_input(uv);
    var blur = vec4f(0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            blur += sample_input(uv + vec2f(f32(i), f32(j)) * e);
        }
    }
    blur /= 9.0;
    var o = v + (v - blur) * p.intensity;
    if (p.punchthrough_alpha != 0) { o.a = v.a; }
    store(c, o);
}
