@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var q = uv_at(c);
    let o = select(select(0.5, 0.0, p.align == 1), 1.0, p.align == 2);
    if (p.axis == 0) { q.x -= p.amount * (q.y - o); } else { q.y -= p.amount * (q.x - o); }
    if (off_image(q)) { store(c, select(vec4f(0.0), gray4(0.0), INPUT_GRAY)); return; }
    store(c, sample_input(q));
}
