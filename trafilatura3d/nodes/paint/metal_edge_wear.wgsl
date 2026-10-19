fn effect(c: vec2i) -> vec4f {
    let curv = load_curvature(c).r;
    let ao = load_baked_ambient_occlusion(c).r;
    // Edges: the convex half of the curvature, softened by Edges Smoothness.
    let edges = smoothstep(0.5 + 0.15 * (1.0 - p.edges_smoothness), 0.8 + 0.2 * p.edges_smoothness, curv) * p.curvature_weight
              + (1.0 - p.curvature_weight) * 0.5;
    let g = grunge(c, p.grunge_scale, p.seed_, p.use_triplanar != 0, p.triplanar_blending_contrast);
    var x = edges - (1.0 - g) * p.grunge_amount * 0.6;
    x *= mix(1.0, ao, p.ambient_occlusion_masking);
    x = contrasted(level_in(x, p.wear_level), p.wear_contrast);
    if (p.invert != 0) { x = 1.0 - x; }
    return vec4f(vec3f(x), 1.0);
}
