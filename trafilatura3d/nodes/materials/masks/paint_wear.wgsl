@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e = max(sample_blurred(uv).r * 1.8, convex(load_curvature(c).r));
    let cl = cells(uv, 20, 1.0, p.seed_);
    var n = sat(cl.z * 0.6 + fbm(uv, 10, 5, 0.6, p.seed_ + 1.0) * 0.7 - 0.15);
    if (p.override_variation_mask != 0) { n = load_variation_mask(c).r; }
    var v = mix(e, e * n * 2.0, p.variation);
    v *= mix(1.0, load_ambient_occlusion(c).r, p.occlusion);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
