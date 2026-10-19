@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let m = sample_m(uv).r;
    let ua = crop_uv(uv);
    let ub = crop_uv(fract(uv + 0.5));
    let ka = select(0.0, m, p.use_base_color != 0);
    let kn = select(0.0, m, p.use_normal != 0);
    let kh = select(0.0, m, p.use_height != 0);
    store_basecolor(c, mix(sample_basecolor(ua), sample_basecolor(ub), ka));
    let na = n_dec(sample_normal(ua).rgb, false);
    let nb = n_dec(sample_normal(ub).rgb, false);
    store_normal(c, vec4f(n_enc(n_lerp(na, nb, kn), false), 1.0));
    store_height(c, g4(mix(sample_height(ua).r, sample_height(ub).r, kh)));
}
