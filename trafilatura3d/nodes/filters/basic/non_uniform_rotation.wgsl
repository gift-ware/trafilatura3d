@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let turns = load_rotation_map(c).r * p.rotation_angle_multiplier + p.rotation_angle_offset;
    var pivot = p.pivot_position;
    let pm = load_rotation_pivot_position_map(c).rg;
    if (p.use_pivot_position_map != 0) { pivot = pm; }
    let s = square_scale();
    let q = pivot + rot((uv - pivot) * s, -turns) / s;
    if (off_image(q)) { store(c, background(p.background_color, INPUT_GRAY)); return; }
    if (p.filtering_mode == 0) { store(c, texel_input(vec2i(floor(q * vec2f(size_input()))))); }
    else { store(c, sample_input(q)); }
}
