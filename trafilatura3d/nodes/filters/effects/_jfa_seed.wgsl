// Jump flooding seeds: pixels above the threshold know themselves as nearest "white" (xy), the
// others as nearest "black" (zw); -1 = none known yet. Stored packed (_jfa_lib.wgsl: 16 bit integers).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = load_src(c);
    var v = t.r;
    if (p.channel == 1) { v = t.a; }
    else if (!SRC_GRAY) { v = luminance(t.rgb); }
    let me = vec2f(c);
    if (v >= p.threshold) { store_seeds(c, jfa_pack(vec4f(me, -1.0, -1.0))); }
    else { store_seeds(c, jfa_pack(vec4f(-1.0, -1.0, me))); }
}
