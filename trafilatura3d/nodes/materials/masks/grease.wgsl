@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let th = max(p.thickness_threshold, 1e-3);
    let thin = sat((th - load_thickness(c).r) / th + 0.3 * th);
    var n = fbm(uv, 8, 5, 0.6, p.seed_);
    if (p.override_noise != 0) { n = load_noise(c).r; }
    let v = thin * (0.3 + n * 1.2);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
