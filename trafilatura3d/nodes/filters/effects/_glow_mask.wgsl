// Glow: the input's luminance cut at Clear Amount (a ramp over Clear Amount +- 0.01).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = load_input(c);
    let m = select(luminance(t.rgb), t.r, INPUT_GRAY);
    store_out(c, fx_gray(clamp((m - p.clear_amount + 0.01) / 0.02, 0.0, 1.0)));
}
