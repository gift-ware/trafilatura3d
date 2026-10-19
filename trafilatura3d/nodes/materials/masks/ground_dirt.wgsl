@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let h = load_position(c).r + (fbm(uv, 6, 5, 0.6, p.seed_) - 0.5) * 0.15;
    let v = sat(1.0 - h / max(p.dirt_height, 1e-3));
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
