const KIND = $K;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var m = load_greyscale_mask(c).r;
    if (p.color_id_mask != 0) { m = id_mask(load_colorid(c).rgb, p.color.rgb, p.fuzziness, p.padding); }
    let mode = p.$ch_blending_mode;
    var k = sat(m) * p.$ch_opacity;
    if (mode == 7) { k = step(0.5, m) * p.$ch_opacity; }
    let b = load_$ch(c);
    var o = vec4f(mix(b.rgb, blend_mode(mode, b.rgb, $value), k), b.a);
    if (KIND == 2) { o = g4(o.r); }
    store_$ch(c, out_$ch(c, o));
}
