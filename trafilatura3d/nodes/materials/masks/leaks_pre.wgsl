@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    var n = smoothstep(0.4, 0.7, fbm(uv, 8, 4, 0.6, p.seed_));
    if (p.override_variation_mask != 0) { n = load_variation_mask(c).r; }
    let e = convex(load_curvature(c).r);
    store(c, g4(mix(e, e * n * 2.0, p.variation)));
}
