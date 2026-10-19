@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / p.size_;
    let n = f32(max(p.tiling, 1));
    let t = fract(select(uv.y, uv.x, p.rotation == 1) * n);
    let m = clamp(p.position, 1e-4, 1.0 - 1e-4);
    let v = select((1.0 - t) / (1.0 - m), t / m, t < m);
    store(c, vec4f(v, v, v, 1.0));
}
