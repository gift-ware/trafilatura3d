@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let dx = p.normal_format == 0;
    let m = read_mat(c, dx);
    let ao = load_baked_ambient_occlusion(c).r;
    let up = sat(wsn_dec(load_baked_normal_world_space(c).rgb).y);
    let nz = fbm(uv, 8, 5, 0.6, p.seed_);
    var col = m.basecolor;
    var rough = m.roughness;
    var nrm = m.normal;
    // Age: tiling wear limited to open areas, with creases; desaturated and brightened.
    let scr = edge_scratches(uv, p.sharp_edges_scratches_scale, p.sharp_edges_scratches_warp_intensity, p.seed_ + 5.0);
    let crease = scratches(uv, 12, 2.5, 1.2, 0.6, 0.04, p.seed_ + 7.0) * p.age_creases;
    let age_k = sat(hscan(nz, p.age, 0.5) + crease) * smoothstep(p.age_threshold - 0.2, p.age_threshold + 0.2, ao) * (1.0 - scr * 0.7);
    let old = saturate_color(col, 1.0 - p.old_fabric_desaturation) * (0.5 + p.old_fabric_brightness);
    col = mix(col, old, age_k);
    rough = mix(rough, min(rough + 0.15, 1.0), age_k);
    // Edges of the relief: sharpened, lighter.
    let e = sat(length(m.normal.xy) * 2.0) * p.edges_wearing;
    nrm = normalize(vec3f(m.normal.xy * (1.0 + p.edges_wearing * 1.5), m.normal.z));
    col = mix(col, saturate_color(col, 0.7) * 1.25, e);
    // Dirtiness, used crease dirt, dust.
    let dirt_k = sat((1.0 - ao) * 1.5 + (nz - 0.5) * 0.6) * p.dirtiness;
    col = mix(col, col * vec3f(0.55, 0.5, 0.45), dirt_k);
    let used_k = sat(pow(1.0 - ao, 2.0) * 3.0) * p.used * (1.0 - scr * 0.7);
    col = mix(col, col * 0.25, used_k);
    rough = mix(rough, 1.0, dirt_k * 0.5);
    let dust_k = up * p.dust * (0.6 + 0.6 * fbm(uv, 16, 4, 0.6, p.seed_ + 3.0));
    col = mix(col, vec3f(0.55, 0.52, 0.47), sat(dust_k));
    rough = mix(rough, 0.95, sat(dust_k));
    var fx = layer(m, col, rough, m.metallic);
    fx.normal = nrm;
    fx.ao = m.ao * (1.0 - used_k * 0.3);
    fx.height = m.height - used_k * 0.02 + dust_k * 0.01;
    var k = 1.0;
    if (p.mask != 0) { k = load_mask(c).r; }
    let w = Blend(p.diffuse_intensity, p.base_color_intensity, p.normal_intensity, p.specular_intensity,
                  p.glossiness_intensity, p.roughness_intensity, 0.0, p.ambient_occlusion_intensity, p.height_intensity);
    write_mat(c, blend_mat(m, fx, w, k), dx);
}
