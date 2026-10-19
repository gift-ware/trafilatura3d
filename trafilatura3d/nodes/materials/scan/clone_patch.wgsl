@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let m = sample_m(uv).r;
    let t = sample_input(uv);
    let s = sample_input(src_uv(uv));
    var o = mix(t, s, m);
    if (!INPUT_GRAY && p.is_normal != 0) { o = vec4f(n_enc(n_lerp(n_dec(t.rgb, false), n_dec(s.rgb, false), m), false), 1.0); }
    store_output(c, o);
    store_areas(c, patch_areas(t, uv));
}
