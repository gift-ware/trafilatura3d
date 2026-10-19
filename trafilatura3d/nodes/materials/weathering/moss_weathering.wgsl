fn moss_at(uv: vec2f) -> f32 {
    let place = sample_baked_position(uv).g * 0.4 + (1.0 - sample_baked_height(uv).r) * 0.4;
    let n = fbm(uv, 6, 6, 0.6, p.seed_);
    return hscan(place + n * 0.6 - 0.2, p.moss_propagation, 0.55);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e1 = 1.0 / vec2f(d);
    let dx = p.normal_format == 0;
    let m = read_mat(c, dx);
    let k0 = moss_at(uv);
    // Fuzzy relief: fine noise grown with the thickness.
    let f0 = vnoise(uv, vec2i(128), p.seed_ + 1.0) * k0;
    let fx_ = vnoise(uv + vec2f(e1.x, 0.0), vec2i(128), p.seed_ + 1.0) * moss_at(uv + vec2f(e1.x, 0.0));
    let fy_ = vnoise(uv - vec2f(0.0, e1.y), vec2i(128), p.seed_ + 1.0) * moss_at(uv - vec2f(0.0, e1.y));
    let thick = smoothstep(0.3, 1.0, k0);
    let col = p.moss_color.rgb * mix(1.0, 0.35, thick) * (0.8 + 0.4 * f0);
    var fx = layer(m, col, 0.95, 0.0);
    fx.normal = normalize(n_lerp(m.normal, vec3f(0.0, 0.0, 1.0), thick) + relief_normal(f0, fx_, fy_, 2.0 * f32(d.x) / 256.0) - vec3f(0.0, 0.0, 1.0));
    fx.height = m.height + thick * 0.08 + f0 * 0.02;
    fx.ao = m.ao * (1.0 - thick * 0.2);
    var k = smoothstep(0.0, 0.3, k0);
    if (p.mask != 0) { k *= load_mask(c).r; }
    let w = Blend(p.diffuse_intensity, p.base_color_intensity, p.normal_intensity, p.specular_intensity,
                  p.glossiness_intensity, p.roughness_intensity, 1.0, p.ambient_occlusion_intensity, p.height_intensity);
    write_mat(c, blend_mat(m, fx, w, k), dx);
}
