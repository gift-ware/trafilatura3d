// Edge wear, the edge mask before the blur: full at half the convex range, scaled by Curvature Weight.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store(c, g4(sat(2.0 * convex(load_curvature(c).r) * p.curvature_weight)));
}
