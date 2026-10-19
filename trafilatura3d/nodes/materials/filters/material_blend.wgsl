const KIND = $K;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var m = load_greyscale_mask(c).r;
    if (p.color_id_mask != 0) { m = id_mask(load_colorid(c).rgb, p.color.rgb, p.fuzziness, p.padding); }
    var k = sat(m) * p.$ch_opacity;
    let mode = $mode;
    if (mode == 7) { k = step(0.5, m) * p.$ch_opacity; }
    let b = load_$ch_bg(c);
    let f = load_$ch_fg(c);
    var o: vec4f;
    if (KIND == 1) {
        o = vec4f(n_enc(n_lerp(n_dec(b.rgb, false), n_dec(f.rgb, false), k), false), 1.0);
    } else {
        o = vec4f(mix(b.rgb, blend_mode(mode, b.rgb, f.rgb), k), mix(b.a, f.a, k));
    }
    if (KIND == 2) { o = g4(o.r); }
    store_$ch(c, out_$ch(c, o));
}
