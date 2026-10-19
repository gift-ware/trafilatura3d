// Final pass: the reduced result upsampled and sharpened.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = sample_state(fx_uv(c)).r;
    let k = 1.0 + 2.0 * p.contrast;
    store(c, fx_gray(clamp((v - 0.5) * k + 0.5, 0.0, 1.0)));
}
