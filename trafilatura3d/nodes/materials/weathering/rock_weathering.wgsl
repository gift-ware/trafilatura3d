fn rock_cracks(uv: vec2f) -> f32 { return crack_pattern(uv, p.cracks_scale, 0.6, p.seed_) * p.cracks_intensity; }

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
    let cv = convex_of(load_baked_curvature(c).r);
    let up = sat(wsn_dec(load_baked_normal_ws(c).rgb).y);
    let nz = fbm(uv, 6, 5, 0.6, p.seed_ + 1.0);
    var col = mix(m.basecolor, saturate_color(m.basecolor, 1.0 - p.used_rock_desaturation) * (0.5 + p.used_rock_brightness),
                  p.used_rock * (0.5 + 0.5 * nz));
    let scr = edge_scratches(uv, p.sharp_edges_scratches_scale, p.sharp_edges_scratches_warp_intensity, p.seed_ + 5.0) * cv;
    let chip = sat(cv * 1.5 * (0.5 + nz)) * p.edges_wearing;
    col = mix(col, col * 1.35 + 0.05, sat(chip + scr * 0.5));
    let age_k = p.age * smoothstep(p.age_threshold + 0.2, p.age_threshold - 0.2, ao) * (0.5 + 0.5 * fbm(uv, 10, 4, 0.6, p.seed_ + 4.0));
    col = mix(col, col * vec3f(0.45, 0.48, 0.4), age_k);
    let dirt_k = sat((1.0 - ao) * 1.5) * p.dirtiness;
    col = mix(col, col * 0.5, dirt_k);
    let dust_k = sat(up * p.dust * (0.6 + 0.6 * fbm(uv, 16, 4, 0.6, p.seed_ + 3.0)));
    col = mix(col, vec3f(0.6, 0.57, 0.52), dust_k);
    let k0 = rock_cracks(uv);
    let kx = rock_cracks(uv + vec2f(e1.x, 0.0));
    let ky = rock_cracks(uv - vec2f(0.0, e1.y));
    col *= 1.0 - k0 * 0.5;
    var fx = layer(m, col, sat(m.roughness + dust_k * 0.2 + k0 * 0.2 - chip * 0.1), m.metallic);
    let strength = p.normal_intensity * f32(d.x) / 256.0 * 0.2;  // default 5
    fx.normal = n_combine(m.normal, relief_normal(-k0 - chip * 0.3, -kx - chip * 0.3, -ky - chip * 0.3, strength));
    fx.height = m.height - k0 * 0.1 - chip * 0.03;
    fx.ao = m.ao * (1.0 - k0 * 0.5);
    var k = 1.0;
    if (p.mask != 0) { k = load_mask(c).r; }
    let w = Blend(p.diffuse_intensity, p.base_color_intensity, min(p.normal_intensity, 1.0), p.specular_intensity,
                  p.glossiness_intensity, p.roughness_intensity, 0.0, p.ambient_occlusion_intensity, p.height_intensity);
    write_mat(c, blend_mat(m, fx, w, k), dx);
}
