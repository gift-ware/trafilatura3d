@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let e = sample_blurred(uv_of(c, d)).r;
    let v = sat((e * 1.6 + (load_cloth_height(c).r - 0.5) * 1.2 - 0.35) * 3.0);
    store(c, g4(v * load_mask(c).r));
}
