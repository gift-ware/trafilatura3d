const KIND = $K;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let m = load_m(c).r;
    let b = load_$ch_bg(c);
    var f = load_$ch_fg(c);
    // Albedo matching: the top's colors scaled towards the bottom's average (1 when not a color channel).
    let ratio = load_mean_bg(c).rgb / max(load_mean_fg(c).rgb, vec3f(1e-3));
    f = vec4f(f.rgb * mix(vec3f(1.0), ratio, p.albedo_matching), f.a);
    var o: vec4f;
    if (KIND == 1) {
        o = vec4f(n_enc(n_lerp(n_dec(b.rgb, false), n_dec(f.rgb, false), m), false), 1.0);
    } else {
        o = mix(b, f, m);
    }
    if (KIND == 2) { o = g4(o.r); }
    store_$ch(c, out_$ch(c, o));
}
