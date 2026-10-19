@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let w = mix(1.0 - p.top_stretch, 1.0 - p.bottom_stretch, uv.y);
    if (abs(w) < 1e-5) { store(c, background(p.background_color, INPUT_GRAY)); return; }
    let q = vec2f(0.5 + (uv.x - 0.5) / w, uv.y);
    if (off_image(q)) { store(c, background(p.background_color, INPUT_GRAY)); return; }
    if (p.sampling == 1) { store(c, texel_input(vec2i(floor(q * vec2f(size_input()))))); }
    else { store(c, sample_input(q)); }
}
