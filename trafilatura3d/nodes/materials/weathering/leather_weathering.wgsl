fn worn_at(uv: vec2f, ao: f32) -> f32 {
    let cr = crack_pattern(uv, p.cracks_scale * 3.0, p.cracks_warp_intensity, p.seed_);
    let global = hscan(fbm(uv, 6, 5, 0.6, p.seed_ + 1.0), p.used, 0.5);
    let creases = p.age * smoothstep(p.age_threshold + 0.15, p.age_threshold - 0.15, ao);
    return sat(max(global, creases) * (0.6 + 0.6 * cr));
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
    let ao = load_baked_ambient_occlusion(c).r;
    let up = sat(wsn_dec(load_baked_normal_world_space(c).rgb).y);
    let scr = edge_scratches(uv, p.sharp_edges_scratches_scale, p.sharp_edges_scratches_warp_intensity, p.seed_ + 5.0);
    let w0 = worn_at(uv, ao) * (1.0 - scr * 0.5);
    let cr0 = crack_pattern(uv, p.cracks_scale * 3.0, p.cracks_warp_intensity, p.seed_);
    let crx = crack_pattern(uv + vec2f(e1.x, 0.0), p.cracks_scale * 3.0, p.cracks_warp_intensity, p.seed_);
    let cry = crack_pattern(uv - vec2f(0.0, e1.y), p.cracks_scale * 3.0, p.cracks_warp_intensity, p.seed_);
    var col = mix(m.basecolor, saturate_color(m.basecolor, 1.0 - p.used_leather_desaturation) * (0.5 + p.used_leather_brightness), w0);
    var nrm = n_combine(m.normal, relief_normal(-cr0, -crx, -cry, w0 * 2.0 * f32(d.x) / 256.0));
    let e = sat(length(m.normal.xy) * 2.0) * p.edges_wearing;
    nrm = normalize(vec3f(nrm.xy * (1.0 + p.edges_wearing), nrm.z));
    col = mix(col, col * 1.3, e);
    var rough = mix(m.roughness, min(m.roughness + 0.2, 1.0), w0);
    let dirt_k = sat((1.0 - ao) * 1.5 + (fbm(uv, 8, 4, 0.6, p.seed_ + 2.0) - 0.5) * 0.6) * p.dirtiness;
    col = mix(col, col * vec3f(0.5, 0.45, 0.4), dirt_k);
    let dust_k = sat(up * p.dust * (0.6 + 0.6 * fbm(uv, 16, 4, 0.6, p.seed_ + 3.0)));
    col = mix(col, vec3f(0.55, 0.52, 0.47), dust_k);
    rough = mix(rough, 0.95, dust_k);
    var fx = layer(m, col, rough, m.metallic);
    fx.normal = nrm;
    fx.height = m.height - cr0 * w0 * 0.03;
    fx.ao = m.ao * (1.0 - dirt_k * 0.2);
    var k = 1.0;
    if (p.mask != 0) { k = load_mask(c).r; }
    let w = Blend(p.diffuse_intensity, p.base_color_intensity, p.normal_intensity, p.specular_intensity,
                  p.glossiness_intensity, p.roughness_intensity, 0.0, p.ambient_occlusion_intensity, p.height_intensity);
    write_mat(c, blend_mat(m, fx, w, k), dx);
}
