fn select_edge(x: f32, width: f32, soft: f32) -> f32 {
    let t = 1.0 - width;
    let s = max(soft, 0.01) * 0.5;
    return smoothstep(t - s, t + s, x);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let curv = load_curvature(c).r;
    let a = select_edge(convex(curv), p.convex_edges_width, p.convex_softness) * p.convex_intensity;
    let b = select_edge(concave(curv), p.concave_edges_width, p.concave_softness) * p.concave_intensity;
    let v = contrast_level(max(a, b), p.contrast * 0.9, p.level);
    store(c, g4(v * load_mask(c).r));
}
