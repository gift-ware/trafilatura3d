fn peel_at(uv: vec2f) -> f32 {
    let cr = crack_pattern(uv, p.paint_cracks_scale * 2.0, p.paint_cracks_warp_intensity, p.seed_);
    let patches = hscan(fbm(uv, 5, 5, 0.6, p.seed_ + 1.0) * 0.8 + cr * 0.3, p.paint_peeling, 0.8);
    return sat(patches);
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
    let up = sat(wsn_dec(load_baked_normal_ws(c).rgb).y);
    // Bare metal: peeled paint, worn edges, scratches.
    let edge = sat(length(m.normal.xy) * 2.5) * p.edges_wearing;
    let scr = edge_scratches(uv, p.sharp_edges_scratches_scale, p.sharp_edges_scratches_warp_intensity, p.seed_ + 5.0)
        * p.edges_wearing;
    let pl = peel_at(uv);
    let bare = sat(max(max(pl, edge), scr));
    var fx = m;
    fx.basecolor = mix(m.basecolor, p.raw_metal_color.rgb, bare);
    fx.diffuse = mix(m.diffuse, vec3f(0.0), bare);
    fx.specular = mix(m.specular, p.raw_metal_specular_color.rgb, bare);
    fx.roughness = mix(m.roughness, p.raw_metal_roughness_value, bare);
    fx.glossiness = mix(m.glossiness, p.raw_metal_glossiness_value, bare);
    fx.metallic = mix(m.metallic, 1.0, bare);
    // Paint thickness: the peeled areas sit lower, with a relief at their border.
    let plx = peel_at(uv + vec2f(e1.x, 0.0));
    let ply = peel_at(uv - vec2f(0.0, e1.y));
    let strength = p.normal_intensity * f32(d.x) / 256.0 * 0.1;  // default 5
    fx.normal = n_combine(m.normal, relief_normal(-pl, -plx, -ply, strength));
    fx.height = m.height - pl * 0.03;
    // Rust / verdigris in occluded areas and on bare metal, flaking.
    let rn = fbm(uv, 8, 5, 0.6, p.seed_ + 2.0);
    let rust_k = hscan((1.0 - ao) * 0.6 + bare * 0.4 + rn * 0.4, p.rust, 0.6);
    let flakes = smoothstep(0.5, 0.3, cells(uv, 24, 1.0, p.seed_ + 6.0).x) * p.rust_peeling;
    var rc = mix(vec3f(0.42, 0.2, 0.08), vec3f(0.6, 0.33, 0.12), rn);
    if (p.rust_verdigris == 1) { rc = mix(vec3f(0.25, 0.55, 0.45), vec3f(0.4, 0.7, 0.55), rn); }
    let rust = layer(fx, rc * (1.0 - flakes * 0.4), 0.85, 0.0);
    fx = blend_mat(fx, rust, Blend(1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0), rust_k);
    fx.height += rust_k * 0.01 * (1.0 + flakes);
    // Dirt and dust.
    let dirt_k = sat((1.0 - ao) * 1.5) * p.dirtiness;
    fx.basecolor = mix(fx.basecolor, fx.basecolor * 0.45, dirt_k);
    fx.diffuse = mix(fx.diffuse, fx.diffuse * 0.45, dirt_k);
    let dust_k = sat(up * p.dust * (0.6 + 0.6 * fbm(uv, 16, 4, 0.6, p.seed_ + 3.0)));
    fx = blend_mat(fx, layer(fx, vec3f(0.55, 0.52, 0.47), 0.95, 0.0), Blend(1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0), dust_k);
    var k = 1.0;
    if (p.mask != 0) { k = load_mask(c).r; }
    let w = Blend(p.diffuse_intensity, p.base_color_intensity, min(p.normal_intensity, 1.0), p.specular_intensity,
                  p.glossiness_intensity, p.roughness_intensity, p.metallic_intensity, p.ambient_occlusion_intensity,
                  p.height_intensity);
    write_mat(c, blend_mat(m, fx, w, k), dx);
}
