// Edge Detect: the edges grown by Edge Width - 1 and by the closing's Edge Roundness (pixels at 256).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = fx_px();
    let reach = (max(p.edge_width - 1.0, 0.0) + ceil(max(p.edge_roundness, 0.0))) * s;
    store_out(c, fx_gray(select(0.0, 1.0, max(shape_sd(c), 0.0) <= reach)));
}
