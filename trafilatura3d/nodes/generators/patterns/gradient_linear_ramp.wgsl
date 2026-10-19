@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / p.size_;
    let n = f32(max(p.tiling, 1));
    // 0: white at the top, each step turns it 90 degrees clockwise.
    var t = fract(select(uv.y, uv.x, (p.rotation & 1) == 1) * n);
    if (p.rotation == 0 || p.rotation == 3) { t = 1.0 - t; }
    let v = t;
    store(c, vec4f(v, v, v, 1.0));
}
