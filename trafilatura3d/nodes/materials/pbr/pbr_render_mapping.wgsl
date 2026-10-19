@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let u = load_uvs(c);
    var bg = p.background_color;
    if (OUT_OUTPUT_GRAY) { bg = g4(bg.r); }
    let v = sample_texture(fract(u.rg));
    store(c, mix(bg, v, sat(u.b)));
}
