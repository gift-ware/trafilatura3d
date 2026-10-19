@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let e = 1.0 / vec2f(size_gradient_input());
    let gx = sample_gradient_input(uv + vec2f(e.x, 0.0)).r - sample_gradient_input(uv - vec2f(e.x, 0.0)).r;
    let gy = sample_gradient_input(uv + vec2f(0.0, e.y)).r - sample_gradient_input(uv - vec2f(0.0, e.y)).r;
    // The slope in uv (change across the image); sampling uphill moves the content downhill, away from white.
    let slope = vec2f(gx, gy) * 0.5 * vec2f(size_gradient_input());
    let q = uv + slope * p.pixel_size_ * p.intensity / 256.0;
    if (p.input_filtering_mode == 0) {
        store(c, texel_input(vec2i(floor(q * vec2f(size_input())))));
    } else {
        store(c, sample_input(q));
    }
}
