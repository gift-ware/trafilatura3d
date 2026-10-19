// Final pass: the grown edges eroded back by Edge Roundness (a closing: rounded inner corners), inverted
// when asked; a pixel of antialiasing.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let r = ceil(max(p.edge_roundness, 0.0)) * fx_px();
    var v = clamp(-shape_sd(c) - r + 0.5, 0.0, 1.0);
    if (p.invert == 0) { v = 1.0 - v; }  // black edges on white; Invert gives white edges
    store(c, fx_gray(v));
}
