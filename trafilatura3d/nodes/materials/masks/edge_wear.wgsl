@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let k = p.threshold * 0.8;
    var e = sat((convex(load_curvature(c).r) - k) / (1.0 - k));
    e = pow(e, mix(3.0, 0.3, p.edges_width));
    let n = fbm(uv, 12, 5, 0.6, p.seed_);
    let v = mix(e, e * (n * 2.2 - 0.1), p.disorder);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
