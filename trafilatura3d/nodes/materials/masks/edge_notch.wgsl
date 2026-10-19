@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let n = fbm(uv, 32, 3, 0.6, p.seed_) * 0.7 + vnoise(uv, vec2i(128), p.seed_ + 1.0) * 0.3;
    let v = sample_blurred(uv).r * (0.3 + n);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
