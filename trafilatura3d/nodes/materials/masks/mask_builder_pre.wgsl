// Curvature layer before smoothing: convex and concave selections, each with its range and contrast.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let curv = load_curvature(c).r;
    var cx = convex(curv);
    if (p.convex_invert != 0) { cx = 1.0 - cx; }
    var cc = concave(curv);
    if (p.concave_invert != 0) { cc = 1.0 - cc; }
    let a = hscan(cx, 0.5 + 0.5 * p.convex_range, p.convex_contrast);
    let b = hscan(cc, 0.5 + 0.5 * p.concave_range, p.concave_contrast);
    store(c, g4(max(a, b)));
}
