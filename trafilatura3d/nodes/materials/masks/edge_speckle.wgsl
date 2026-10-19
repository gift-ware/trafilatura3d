@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let curv = load_curvature(c).r;
    let e = mix(abs(curv - 0.5) * 2.0, convex(curv), p.edge_selection);
    let s = cells(uv, 48, 1.0, p.seed_);
    let speck = smoothstep(0.55, 0.25, s.x) * step(0.35, s.z);
    var n = fbm(uv, 3, 5, 0.5, p.seed_ + 2.0);
    if (p.override_variation_mask != 0) { n = load_variation_mask(c).r; }
    let v = e * (0.35 + speck) * mix(1.0, n * 2.0, p.variation);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
