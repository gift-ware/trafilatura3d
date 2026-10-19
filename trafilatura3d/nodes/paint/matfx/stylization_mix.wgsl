// Stylization: the global mix of the stylized channels over the material (a helper is shown as is).
fn fx(c: vec2i) {
    let k = select(p.stylization, 1.0, p.display_helpers != 0);
    store_basecolor(c, mix(load_basecolor(c), load_s_basecolor(c), k));
    store_roughness(c, mix(load_roughness(c), load_s_roughness(c), p.stylization));
    store_metallic(c, mix(load_metallic(c), load_s_metallic(c), p.stylization));
    let n = mix(mx_ndec(load_normal(c).rgb), mx_ndec(load_s_normal(c).rgb), p.stylization);
    store_normal(c, vec4f(mx_nenc(n), load_normal(c).a));
    store_height(c, mix(load_height(c), load_s_height(c), p.stylization));
    store_ambient_occlusion(c, mix(load_ambient_occlusion(c), load_s_ambient_occlusion(c), p.stylization));
}
