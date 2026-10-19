@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let fdx = p.frost_normal_format == 0;
    let m = read_mat(c, fdx);
    // The Mask lowers the water: the level is Water Level times the mask.
    let level = p.water_level * load_mask(c).r;
    let depth = level - m.height;
    let under = smoothstep(-0.005, 0.005, depth) * step(0.001, level);
    // Wetness: full under water and up to half Edges Wetness Distance above it, then down to dry at 1.45 times
    // the distance; the Mask output.
    let dist = p.edges_wetness_distance;
    let wetness = select(0.0, sat((1.45 * dist + depth) / (0.97 * dist)), dist > 0.0) * p.edges_wetness;
    let wet = wetness * (1.0 - under);  // the shore
    var o = m;
    o.basecolor = mix(m.basecolor, m.basecolor * m.basecolor, wet * 0.7);
    o.diffuse = mix(m.diffuse, m.diffuse * m.diffuse, wet * 0.7);
    o.roughness = mix(m.roughness, m.roughness * 0.3, wet);
    o.glossiness = 1.0 - o.roughness;
    // Under water: blurred, darkened with depth, sludge deep down.
    let dk = sat(depth * 6.0);
    var bc = mix(m.basecolor, sample_bb(uv).rgb, dk * p.depth_blur_opacity);
    var df = mix(m.diffuse, sample_db(uv).rgb, dk * p.depth_blur_opacity);
    let dark = mix(1.0, exp(-depth * 12.0), p.water_darkness);
    let sl = smoothstep(p.sludge_depth * 0.3, p.sludge_depth * 0.3 + 0.05, depth) * p.sludge_opacity;
    bc = mix(bc * dark, p.sludge_color.rgb, sl);
    df = mix(df * dark, p.sludge_color.rgb, sl);
    var w = layer(m, bc, 0.02, 0.0);
    w.diffuse = df;
    w.normal = vec3f(0.0, 0.0, 1.0);
    w.height = level;
    w.ao = mix(m.ao, 1.0, 0.5);
    w.specular_level = 0.5;
    // Frost: from the shore inwards (small depth first), cracked.
    let reach = p.frost * 0.4;
    let fz = smoothstep(reach, reach * 0.6, depth) * step(0.0001, reach) * p.frost_intensity;
    let cr = cells(uv, 10, 1.0, p.seed_);
    let crack = smoothstep(0.06 * p.frost_cracks + 0.001, 0.0, cr.w) * p.frost_cracks;
    let fbmv = fbm(uv, 24, 4, 0.6, p.seed_ + 1.0);
    var fr = layer(m, mix(bc, vec3f(0.8, 0.86, 0.9), 0.75) * (1.0 - crack * 0.4), 0.35 + fbmv * 0.3 + crack * 0.3, 0.0);
    fr.normal = normalize(vec3f((fbmv - 0.5) * 0.4 - crack * 0.3, (fbmv - 0.5) * 0.3 + crack * 0.2, 1.0));
    fr.height = level + 0.005;
    let allw = Blend(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    w = blend_mat(w, fr, allw, fz);
    write_mat(c, blend_mat(o, w, allw, under), fdx);
    store_mask(c, vec4f(wetness, wetness, wetness, 1.0));
}
