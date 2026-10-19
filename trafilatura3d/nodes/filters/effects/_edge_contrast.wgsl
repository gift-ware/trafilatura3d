// Edge Detect, the edge pixels (ours): white where the pixel stands out of the mean of its 4 neighbours (a
// Laplacian: smooth ramps are not edges) by the threshold Tolerance sets (0 = the slightest contrast); a step of
// height h is an edge from h = 4 x threshold on.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let around = load_input(c + vec2i(1, 0)).r + load_input(c - vec2i(1, 0)).r
               + load_input(c + vec2i(0, 1)).r + load_input(c - vec2i(0, 1)).r;
    let contrast = abs(around * 0.25 - load_input(c).r);
    let threshold = mix(0.000075, 0.0002625, p.tolerance);
    store_out(c, fx_gray(select(0.0, 1.0, contrast >= threshold)));
}
