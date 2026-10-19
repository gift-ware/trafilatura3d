@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let per = vec2i(round(f32(max(p.scale, 1)) * asp));
    let x = uv_of(c) * vec2f(per);
    let r = cells2(x, per, p.seed_, 1.0, p.disorder * p.disorder_speed, 0.0, 0.0, 2.0);
    // A hard binary mask (the cell borders grown by a Distance node in proportion to Edge width
    // / Scale): walls 0.15 x Edge width cells wide, fitted on the preview at the default 2.
    var v = select(0.0, 1.0, r.edge >= 0.075 * p.edge_width);
    if (p.invert != 0) { v = 1.0 - v; }
    store(c, vec4f(v, v, v, 1.0));
}
