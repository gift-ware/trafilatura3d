@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let m = sample_m(uv).r;
    let su = src_uv(uv);
    var t: vec4f;
    var s: vec4f;
    var o: vec4f;
//@each
    t = sample_$ch(uv);
    s = sample_$ch(su);
    o = mix(t, s, m);
    if (!$CH_GRAY && p.is_normal != 0) { o = vec4f(n_enc(n_lerp(n_dec(t.rgb, false), n_dec(s.rgb, false), m), false), 1.0); }
    if ($i >= p.input_count) { o = vec4f(0.0, 0.0, 0.0, 1.0); }
    store_output_$n(c, o);
//@end
    store_areas(c, patch_areas(sample_input_1(uv), uv));
}
