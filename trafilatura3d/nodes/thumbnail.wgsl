@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let src = vec2f(size_image());
    let ratio = src / vec2f(d);
    let n = clamp(vec2i(ceil(ratio)), vec2i(1), vec2i(8));
    var acc = vec4f(0.0);
    for (var j = 0; j < n.y; j++) {
        for (var i = 0; i < n.x; i++) {
            let q = (vec2f(c) + (vec2f(f32(i), f32(j)) + 0.5) / vec2f(n)) * ratio;
            acc += texel_image(vec2i(floor(q)));
        }
    }
    store(c, clamp(acc / f32(n.x * n.y), vec4f(0.0), vec4f(1.0)));
}
