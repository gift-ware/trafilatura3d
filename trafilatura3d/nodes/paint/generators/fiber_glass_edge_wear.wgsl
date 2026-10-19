// Fibers: thin ridges of value noise stretched along three random directions of a plane, strongest one wins.
fn fibers2(q: vec2f) -> f32 {
    var f = 0.0;
    for (var k = 0; k < 3; k++) {
        let a = hash21(vec2i(k, 7), p.seed_) * PI;
        let r = g_rot(q, a / (2.0 * PI));
        f = max(f, pow(g_ridged(vec3f(r.x * 6.0, r.y * 90.0, f32(k) * 3.1), p.seed_ + 5.0), 6.0));
    }
    return f;
}

// In UV space, or projected from the three axes like the grunge.
fn fibers(c: vec2i) -> f32 {
    if (p.use_triplanar == 0) { return fibers2(uv_of(c) * 4.0); }
    let pos = load_position(c).rgb * 4.0;
    let n = abs(g_normal(load_world_space_normals(c)));
    var w = pow(n, vec3f(1.0 + p.triplanar_blending_contrast * 16.0));
    w /= max(w.x + w.y + w.z, 1e-6);
    return w.x * fibers2(pos.zy) + w.y * fibers2(pos.xz) + w.z * fibers2(pos.xy);
}

fn effect(c: vec2i) -> vec4f {
    let mc = micro_curvature(c, p.micro_normal, p.micro_height, p.curvature_type, p.curvature_intensity,
                             p.height_details_intensity);
    let curv = g_invert(clamp(load_curvature(c).r + mc - 0.5, 0.0, 1.0), p.invert);
    let ao = g_invert(load_baked_ambient_occlusion(c).r * micro_occlusion(c, p.micro_normal, p.micro_height,
                      p.curvature_type, p.curvature_intensity, p.height_details_intensity, p.ao_radius, p.ao_depth),
                      p.invert);
    let edges = smoothstep(0.5 + 0.15 * (1.0 - p.edges_smoothness), 0.8 + 0.2 * p.edges_smoothness, curv)
                * p.curvature_weight + (1.0 - p.curvature_weight) * 0.5;
    var g = grunge(c, 1.0, p.seed_, p.use_triplanar != 0, p.triplanar_blending_contrast);
    let custom_ = tex_custom_grunge(c, 1.0, p.use_triplanar, p.triplanar_blending_contrast, 0, 1, 0, 0.0, 0.0);
    if (p.use_custom_grunge != 0 && custom_.y > 0.0) { g = custom_.x; }
    var x = edges - (1.0 - g) * p.grunge_amount * 0.6;
    // Fraying: fibers reach out of the wear and are torn out of it along its border.
    let band = smoothstep(0.0, 0.25, x + 0.15) * (1.0 - smoothstep(0.45, 0.8, x));
    x += (fibers(c) - 0.3) * 0.6 * band;
    x *= mix(1.0, ao, p.ambient_occlusion_masking);
    x = contrasted(level_in(x, p.wear_level), p.wear_contrast);
    return vec4f(vec3f(x), 1.0);
}
