const KIND = $K;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var m = load_greyscale_mask(c).r;
    if (p.color_id_mask != 0) { m = id_mask(load_color_id_mask(c).rgb, p.color.rgb, p.fuzziness, p.padding); }
    m = sat(m);
    let v = load_$ch(c);
    var o = v;
    if (KIND == 1) {
        let n = n_dec(v.rgb, false);
        o = vec4f(n_enc(n_lerp(n, vec3f(0.0, 0.0, 1.0), (1.0 - p.intensity) * m), false), 1.0);
    } else {
        var a = contrast_luminosity(v.rgb, $con, $lum);
        if (KIND == 0) { a = hsl_adjust(a, $hue, $sat); }
        let mode = $mode;
        var k = m * $opacity;
        if (mode == 7) { k = step(0.5, m) * $opacity; }
        o = vec4f(mix(v.rgb, blend_mode(mode, v.rgb, a), k), v.a);
        if (KIND == 2) { o = g4(o.r); }
    }
    store_$ch(c, out_$ch(c, o));
}
