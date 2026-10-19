@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let curv = load_curvature(c).r;
    let e = max(concave(curv), convex(curv) * 0.6);
    var n = fbm(uv, 3, 5, 0.5, p.seed_);
    if (p.override_variation_mask != 0) { n = load_variation_mask(c).r; }
    let v = mix(e, e * n * 2.0, p.variation);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
