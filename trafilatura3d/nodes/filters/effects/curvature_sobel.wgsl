fn nrm(c: vec2i) -> vec3f { return fx_decode_normal(load_input(c), p.input_normal_type == 0); }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var dx = 0.0;
    var dy = 0.0;
    for (var k = -1; k <= 1; k++) {
        let w = select(1.0, 2.0, k == 0);
        dx += w * (nrm(c + vec2i(1, k)).x - nrm(c + vec2i(-1, k)).x);
        dy += w * (nrm(c + vec2i(k, -1)).y - nrm(c + vec2i(k, 1)).y);  // +Y up
    }
    // (Sobel X of red + Sobel Y of green) x Intensity x 0.5 + 0.5, on the encoded normal.
    store(c, fx_gray(clamp(0.5 + 0.25 * (dx + dy) * p.intensity, 0.0, 1.0)));
}
