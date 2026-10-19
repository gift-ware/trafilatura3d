fn crack_at(uv: vec2f) -> f32 {
    let place = concave_of(sample_baked_curvature(uv).r) * 0.5 + (1.0 - sample_baked_height(uv).r) * 0.5;
    let reveal = hscan(place * 0.6 + fbm(uv, 4, 4, 0.5, p.seed_ + 9.0) * 0.6, p.cracks_propagation, 0.7);
    return crack_pattern(uv, 6.0, 0.8, p.seed_) * reveal;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e = 1.0 / vec2f(d);
    let dx = p.normal_format == 0;
    let m = read_mat(c, dx);
    let k0 = crack_at(uv);
    let kx = crack_at(uv + vec2f(e.x, 0.0));
    let ky = crack_at(uv - vec2f(0.0, e.y));
    var fx = m;
    let depth = p.cracks_depth;
    fx.height = m.height - k0 * depth * 0.3;
    fx.normal = n_combine(m.normal, relief_normal(-k0, -kx, -ky, depth * 4.0 * f32(d.x) / 256.0));
    let dark = 1.0 - k0 * (0.3 + depth * 0.4);
    fx.basecolor = m.basecolor * dark;
    fx.diffuse = m.diffuse * dark;
    fx.roughness = mix(m.roughness, 1.0, k0 * 0.6);
    fx.glossiness = mix(m.glossiness, 0.0, k0 * 0.6);
    fx.ao = m.ao * (1.0 - k0 * depth * 0.6);
    var k = 1.0;
    if (p.mask != 0) { k = load_mask(c).r; }
    let w = Blend(p.diffuse_intensity, p.base_color_intensity, p.normal_intensity, p.specular_intensity,
                  p.glossiness_intensity, p.roughness_intensity, 1.0, p.ambient_occlusion_intensity, p.height_intensity);
    write_mat(c, blend_mat(m, fx, w, k), dx);
}
