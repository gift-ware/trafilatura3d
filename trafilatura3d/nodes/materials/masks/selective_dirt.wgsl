@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    var n = grunge(uv, 4.0, p.seed_);
    if (p.override_variation_mask != 0) { n = load_variation_mask(c).r; }
    let e = convex(load_curvature(c).r);
    let v = mix(e, e * n * 2.0, p.variation);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
