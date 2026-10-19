// Noise that eats into a component (Noisiness): 1 - amount x (1 - fine fbm).
fn noisy(c: vec2i, amount: f32, salt: f32) -> f32 {
    let n = grunge(c, p.grunge_scale * 4.0, p.seed_ + salt, p.use_triplanar != 0, p.triplanar_blending_contrast);
    return 1.0 - amount * (1.0 - n);
}

// Scratches: per cell of a grid (Scale), with probability Amount, a thin random segment; 1 on the segment.
fn scratches(uv: vec2f) -> f32 {
    let g = uv * 6.0 * p.scratches_scale;
    let cell = vec2i(floor(g));
    var s = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let q = cell + vec2i(i, j);
            if (hash21(q, p.seed_ + 11.0) > p.scratches_amount) { continue; }
            let h = hash22(q, p.seed_ + 23.0);
            let centre = vec2f(q) + h;
            let a = hash21(q, p.seed_ + 37.0) * PI;
            let dir = vec2f(cos(a), sin(a));
            let half_len = 0.4 + 0.8 * hash21(q, p.seed_ + 41.0);
            let d = g - centre;
            let t = clamp(dot(d, dir), -half_len, half_len);
            let dist = length(d - dir * t);
            let width = 0.012 + 0.02 * hash21(q, p.seed_ + 53.0);
            s = max(s, smoothstep(width, width * 0.3, dist) * (1.0 - abs(t) / half_len * 0.5));
        }
    }
    return s;
}

// Scatter: per cell (Scale), with probability Density, a stamp (Scatter input, or a soft disc) of Size reduced by
// Size Variation, opacity reduced by Opacity Variation; the strongest stamp wins.
fn scatter(uv: vec2f) -> f32 {
    let g = uv * 8.0 * p.scatter_scale;
    let cell = vec2i(floor(g));
    var s = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let q = cell + vec2i(i, j);
            if (hash21(q, p.seed_ + 61.0) > p.scatter_density) { continue; }
            let centre = vec2f(q) + hash22(q, p.seed_ + 67.0);
            let size = max(p.scatter_size * (1.0 - p.size_variation * hash21(q, p.seed_ + 71.0)), 1e-3);
            let opacity = 1.0 - p.opacity_variation * hash21(q, p.seed_ + 73.0);
            let local = (g - centre) / size;  // the stamp spans -1..1
            if (any(abs(local) > vec2f(1.0))) { continue; }
            var v = smoothstep(1.0, 0.6, length(local));
            if (SCATTER_INPUT_CONNECTED) { v = sample_scatter_input(local * 0.5 + 0.5).r; }
            s = max(s, v * opacity);
        }
    }
    return s;
}

fn effect(c: vec2i) -> vec4f {
    let uv = uv_of(c);
    // Grunge: procedural, or the custom grunges (the second multiplied in at its visibility).
    var g = grunge(c, p.grunge_scale, p.seed_, p.use_triplanar != 0, p.triplanar_blending_contrast);
    let g1 = tex_custom_grunge_1(c, p.grunge_scale, p.use_triplanar, p.triplanar_blending_contrast, 0, 1, 0, 0.0, 0.0);
    let g2 = tex_custom_grunge_2(c, p.grunge_scale, p.use_triplanar, p.triplanar_blending_contrast, 0, 1, 0, 0.0, 0.0);
    if (p.use_custom_grunge != 0 && g1.y > 0.0) { g = g1.x; }
    g = mix(g, g * g2.x, p.secondary_custom_grunge * g2.y);
    g = g_invert(g, p.grunge_invert);
    // Ambient occlusion: occluded white within Range.
    let ao = load_baked_ambient_occlusion(c).r * micro_occlusion(c, p.micro_normal, p.micro_height, p.curvature_type,
                                                                 p.curvature_intensity, p.height_details_intensity,
                                                                 p.ao_radius, p.ao_depth);
    var a = clamp((1.0 - ao) / max(p.ao_range, 1e-3), 0.0, 1.0);
    a = g_invert(g_contrast(a, p.ao_contrast), p.ao_invert) * noisy(c, p.ao_noisiness, 101.0);
    // Curvature: convex and concave parts above the angle their Range sets (x Level Boost), soft by Smoothness.
    let cv = clamp(load_curvature(c).r + micro_curvature(c, p.micro_normal, p.micro_height, p.curvature_type,
                                                         p.curvature_intensity, p.height_details_intensity) - 0.5, 0.0, 1.0);
    let w = 0.02 + p.smoothness * 0.3;
    let tx = 1.0 - clamp(p.convex_range * p.level_boost, 0.0, 1.0);
    let tc = 1.0 - clamp(p.concave_range * p.level_boost, 0.0, 1.0);
    var convex = smoothstep(tx - w, tx + w, (cv - 0.5) * 2.0);
    var concave = smoothstep(tc - w, tc + w, (0.5 - cv) * 2.0);
    convex = g_invert(g_contrast(convex, p.convex_contrast), p.convex_invert);
    concave = g_invert(g_contrast(concave, p.concave_contrast), p.concave_invert);
    let curv = max(convex, concave) * noisy(c, p.curvature_noisiness, 131.0);
    // Top/down gradient: white at the top, reaching down by Range.
    let pos = load_position(c).rgb;
    var grad = smoothstep(1.0 - max(p.gradient_range, 1e-3), 1.0, pos.y);
    grad = g_invert(g_contrast(grad, p.gradient_contrast), p.gradient_invert);
    // World space normal: each facing direction at its intensity.
    let n = g_normal(load_world_space_normals(c));
    let wsn = clamp(max(n.y, 0.0) * p.top_intensity + max(-n.y, 0.0) * p.bottom_intensity
                  + max(n.x, 0.0) * p.front_intensity + max(-n.x, 0.0) * p.back_intensity
                  + max(n.z, 0.0) * p.right_intensity + max(-n.z, 0.0) * p.left_intensity, 0.0, 1.0);
    // Sum: grunge both adds (above its middle) and eats (below); scratches only inside that part.
    var wear = clamp(p.ao * a + p.curvature * curv + p.grunge * (g - 0.5) * 2.0, 0.0, 1.0);
    wear = clamp(wear + p.scratches * scratches(uv) * smoothstep(0.0, 0.25, wear), 0.0, 1.0);
    var m = clamp(wear + p.top_down_gradient * grad + p.world_space_normal * wsn + p.scatter * scatter(uv), 0.0, 1.0);
    m = g_invert(g_contrast(g_balance(m, p.level), p.contrast), p.invert);
    return vec4f(vec3f(m), 1.0);
}
