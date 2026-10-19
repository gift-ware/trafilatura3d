@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let s = square_scale();
    let si = vec2f(size_input());
    let aspect = p.size_.x / p.size_.y;
    let ratio = aspect / (si.x / si.y);
    var rep = vec2f(max(ratio, 1.0), max(1.0 / ratio, 1.0));
    if (p.tile_mode == 1) { rep = vec2f(max(p.tile, vec2i(1))) * vec2f(min(1.0, aspect), min(1.0, 1.0 / aspect)); }
    var turns = p.rotation;
    if (p.safe_rotation_square_only != 0) { turns = floor(turns * 4.0) / 4.0; }
    let offset = floor(p.offset * p.size_) / p.size_;  // whole pixels
    let v = rot((uv - 0.5 - offset) * s, -turns) / s + 0.5;
    if (off_image(v)) { store(c, background(p.background_color, INPUT_GRAY)); return; }
    store(c, sample_input(v * rep));
}
