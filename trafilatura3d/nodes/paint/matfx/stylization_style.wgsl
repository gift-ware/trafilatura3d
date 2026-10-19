// The stylized material (before the global Stylization mix): brush strokes, colorize, gradient, baked lighting,
// edges and cavities on the smoothed channels; or a helper mask in the base color.

fn blend_mode(m: i32, b: vec3f, f: vec3f) -> vec3f {
    switch m {
        case 1: { return b * f; }
        case 2: { return 1.0 - (1.0 - b) * (1.0 - f); }
        case 3: { return select(1.0 - 2.0 * (1.0 - b) * (1.0 - f), 2.0 * b * f, b < vec3f(0.5)); }
        case 4: { return (1.0 - 2.0 * f) * b * b + 2.0 * f * b; }
        case 5: { return b + f; }
        default: { return f; }
    }
}

fn pattern(uv: vec2f) -> f32 {
    let t = max(p.tilling_amount, 1);
    let n = max(i32(round(f32(t) * 3.0 * p.pattern_scale)), 1);
    switch p.grunge_variation {
        case 1: { return mx_cells(uv, vec2i(n), 1.0, p.seed_ + 50.0).z; }
        case 2: { return mx_noise(uv, vec2i(n * 6, n), p.seed_ + 51.0) * 0.7 + mx_noise(uv, vec2i(n * 12, n * 2), p.seed_ + 52.0) * 0.3; }
        case 3: { return smoothstep(0.35, 0.1, mx_cells(uv, vec2i(n * 2), 1.0, p.seed_ + 53.0).x); }
        default: { return mx_fbm(uv, n, 4, 0.55, p.seed_ + 54.0); }
    }
}

fn dir_of(h: f32, v: f32) -> vec3f {
    let a = radians(h);
    let e = radians(v);
    return vec3f(cos(e) * cos(a), sin(e), cos(e) * sin(a));
}

// The world normal, its tangent detail (baked + material normals) applied in a frame built from it.
fn world_normal(c: vec2i, det: vec3f) -> vec3f {
    var n = vec3f(0.0, 0.0, 1.0);
    if (WORLD_SPACE_NORMALS_CONNECTED) { n = mx_ndec(load_world_space_normals(c).rgb); }
    var t = cross(vec3f(0.0, 1.0, 0.0), n);
    if (length(t) < 1e-3) { t = cross(n, vec3f(0.0, 0.0, 1.0)); }
    t = normalize(t);
    let b = cross(n, t);
    return normalize(t * det.x + b * det.y + n * det.z);
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.0 / p.size_;
    let base = load_k_basecolor(c);
    let st = load_strokes(c);
    let cov = st.r * p.brush_strokes;

    // Brush strokes
    var col = base.rgb;
    var sc = col;
    if (p.color_custom != 0) { sc = mix(sc, p.stroke_color.rgb, p.stroke_color_opacity); }
    sc *= 1.0 + (st.g - 0.5) * p.color_variation * 0.7;
    sc = mx_hsl_adjust(sc, (st.b - 0.5) * p.color_random * 0.5, 0.5 + (st.g - 0.5) * p.color_random, 0.5);
    col = mix(col, mx_sat3(sc), cov);
    var rough = load_k_roughness(c).r;
    if (p.roughness_custom != 0) { rough = mix(rough, p.stroke_roughness, cov); }
    rough = mx_sat(rough + (st.b - 0.5) * p.roughness_variation * cov);
    var metal = load_k_metallic(c).r;
    if (p.metallic_custom != 0) { metal = mix(metal, p.stroke_metallic, cov); }
    metal = mx_sat(metal + (st.g - 0.5) * p.metallic_variation * cov);
    var nrm = mx_ndec(load_k_normal(c).rgb);
    let bx = load_strokes(c + vec2i(1, 0)).a - st.a;
    let by = load_strokes(c - vec2i(0, 1)).a - st.a;
    if (p.normal_custom != 0) {
        var sn = mx_nslope(bx, by, 3.0 * mx_px() * p.normal_variation);
        sn = normalize(sn + vec3f(st.g - 0.5, st.b - 0.5, 0.0) * p.normal_random * st.r);
        let k = p.brush_strokes;
        if (p.blending_mode_normal == 0) { nrm = mx_ndetail(nrm, normalize(mix(vec3f(0.0, 0.0, 1.0), sn, k))); }
        else if (p.blending_mode_normal == 1) { nrm = normalize(mix(nrm, sn, cov)); }
        else { nrm = normalize(nrm + (sn - vec3f(0.0, 0.0, 1.0)) * k); }
    }

    // Colorize
    col = mix(col, p.colorize_color.rgb, p.colorize_opacity * p.colorize);
    let g = pattern(uv);
    col = mix(col, col * p.grunge_color.rgb * 1.5, g * p.grunge_opacity * p.colorize);

    // Gradient along a world direction (the position map; image up without it)
    var pos = vec3f(uv.x, 1.0 - uv.y, 0.5);
    if (POSITION_CONNECTED) { pos = load_position(c).rgb; }
    var gm = dot(pos - 0.5, dir_of(p.gradient_horizontal_rotation, p.gradient_vertical_rotation)) + p.gradient_offset;
    gm = mx_contrast(mx_sat(gm), p.gradient_contrast);
    if (p.gradient_invert != 0) { gm = 1.0 - gm; }
    var gcol = col;
    if (p.gradient_mode == 0) {
        gcol = mix(gcol, blend_mode(p.color_blending_mode, gcol, p.gradient_color.rgb), (1.0 - gm) * p.gradient_color_opacity);
        gcol = mix(gcol, blend_mode(p.color_2_blending_mode, gcol, p.color_2.rgb), gm * p.color_2_opacity);
    } else {
        gcol = mix(gcol, blend_mode(p.color_blending_mode, gcol, p.gradient_color.rgb), gm * p.gradient_color_opacity);
    }
    col = mix(col, mx_sat3(gcol), p.gradient);

    // Baked lighting
    var det = mx_ndetail(mx_ndec(load_baked_normal(c).rgb), nrm);
    det = normalize(det + vec3f(st.g - 0.5, st.b - 0.5, 0.0) * p.brush_strokes_in_lighting * cov);
    let wn = world_normal(c, det);
    let l = dir_of(p.light_horizontal_rotation, p.light_vertical_rotation);
    let wrap_k = p.diffuse_radius;
    var dif = mx_sat((dot(wn, l) + wrap_k) / (1.0 + wrap_k));
    let dc = 0.5 * (1.0 - p.diffuse_contrast * 0.95);
    dif = smoothstep(0.5 - dc, 0.5 + dc, dif);
    let hv = normalize(l + vec3f(0.0, 0.0, 1.0));
    var spec = pow(max(dot(wn, hv), 0.0), mix(160.0, 6.0, p.specular_radius));
    let sc2 = 0.5 * (1.0 - p.specular_contrast * 0.95);
    spec = smoothstep(0.5 - sc2, 0.5 + sc2, spec);
    var lit = col * mix(vec3f(1.0), (0.45 + 1.1 * dif) * p.diffuse_color.rgb, p.diffuse_intensity);
    lit += spec * p.specular_color.rgb * p.specular_intensity;
    let ring = (load_k_basecolor(c + vec2i(1, 0)).rgb + load_k_basecolor(c - vec2i(1, 0)).rgb
              + load_k_basecolor(c + vec2i(0, 1)).rgb + load_k_basecolor(c - vec2i(0, 1)).rgb) * 0.25;
    lit += (base.rgb - ring) * p.color_sharpen * 3.0;
    let curv = load_curvature(c).r;
    lit *= 1.0 + (curv - 0.5) * p.surface_sharpen * 1.2 - (1.0 - load_baked_ambient_occlusion(c).r) * p.surface_sharpen * 0.4;
    col = mix(col, mx_sat3(lit), p.baked_lighthing);

    // Edges and cavities (curvature)
    var cav = mx_sat((0.5 - curv) * mix(10.0, 2.0, p.cavities_spread));
    var edg = mx_sat((curv - 0.5) * mix(10.0, 2.0, p.edges_spread));
    cav = mx_contrast(cav, p.edges_and_cavities_contrast) * (1.0 - p.brush_strokes_in_cavities * cov * 0.8);
    edg = mx_contrast(edg, p.edges_and_cavities_contrast) * (1.0 - p.brush_strokes_in_edges * cov * 0.8);
    if (p.mode == 2) { cav = 0.0; }
    if (p.mode == 1) { edg = 0.0; }
    var cav_col = col * vec3f(0.42, 0.38, 0.45);
    if (p.custom_cavities_color != 0) { cav_col = p.cavities_color.rgb; }
    var edg_col = mix(col, vec3f(1.0), 0.35);
    if (p.custom_edges_color != 0) { edg_col = p.edges_color.rgb; }
    col = mix(col, cav_col, cav * p.cavities_opacity * p.edges_and_cavities);
    col = mix(col, edg_col, edg * p.edges_opacity * p.edges_and_cavities);

    // Display helpers
    switch p.display_helpers {
        case 1: { col = vec3f(st.r); }
        case 2: { col = mx_hsv2rgb(vec3f(st.g, 0.7, 1.0)) * st.r; }
        case 3: { col = vec3f(edg, cav, 0.0); }
        case 4: { col = vec3f(gm); }
        case 5: { col = vec3f(dif * 0.8 + spec * 0.2); }
        default: {}
    }

    store_s_basecolor(c, vec4f(mx_sat3(col), base.a));
    let ro = load_k_roughness(c);
    store_s_roughness(c, vec4f(vec3f(rough), ro.a));
    let me = load_k_metallic(c);
    store_s_metallic(c, vec4f(vec3f(metal), me.a));
    store_s_normal(c, vec4f(mx_nenc(nrm), load_k_normal(c).a));
    let h = load_k_height(c);
    store_s_height(c, vec4f(vec3f(h.r + (st.a - 0.5) * cov * 0.02), h.a));
    store_s_ambient_occlusion(c, load_k_ambient_occlusion(c));
}
