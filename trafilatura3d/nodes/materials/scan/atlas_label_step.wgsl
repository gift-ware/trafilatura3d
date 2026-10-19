fn less(a: vec2f, b: vec2f) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let me = texel_lbl(c);
    if (me.x < 0.0) { store(c, me); return; }
    var best = me.xy;
    let offs = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
    for (var i = 0; i < 4; i++) {
        let n = texel_lbl(c + offs[i]);
        if (n.x >= 0.0 && less(n.xy, best)) { best = n.xy; }
    }
    // Pointer jumping: the label of the texel our label points to.
    let j = texel_lbl(vec2i(best));
    if (j.x >= 0.0 && less(j.xy, best)) { best = j.xy; }
    store(c, vec4f(best, 0.0, 1.0));
}
