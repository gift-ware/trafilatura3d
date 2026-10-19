@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let m = sample_m(uv).r * load_mask(c).r;
    let su = src_uv(uv);
    var t: vec4f;
    var s: vec4f;
    var o: vec4f;
//@each
    t = sample_$ch(uv);
    s = sample_$ch(su);
    o = mix(t, s, m);
    if ($K == 1) { o = vec4f(n_enc(n_lerp(n_dec(t.rgb, false), n_dec(s.rgb, false), m), false), 1.0); }
    if ($K == 2) { o = g4(o.r); }
    store_$ch(c, out_$ch(c, o));
//@end
    store_areas_basecolor(c, patch_areas(sample_basecolor(uv), uv));
    store_areas_normal(c, patch_areas(sample_normal(uv), uv));
    store_areas_height(c, patch_areas(g4(sample_height(uv).r), uv));
}
