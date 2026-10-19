fn effect(c: vec2i) -> vec4f {
    let curv = load_curvature(c).r;
    let ao = load_baked_ambient_occlusion(c).r;
    let cavity = clamp((0.5 - curv) * 4.0, 0.0, 1.0);
    let edge = clamp((curv - 0.5) * 4.0, 0.0, 1.0);
    let g = grunge(c, p.grunge_scale, p.seed_, p.use_triplanar != 0, p.triplanar_blending_contrast);
    var x = max(1.0 - ao, cavity) - (1.0 - g) * p.grunge_amount * 0.6;
    x *= 1.0 - edge * p.edges_masking;
    x = contrasted(level_in(x, p.dirt_level), p.dirt_contrast);
    if (p.invert != 0) { x = 1.0 - x; }
    return vec4f(vec3f(x), 1.0);
}
