@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let curv = load_curvature(c).r;
    var v = (1.0 - load_ambient_occlusion(c).r) * 0.9 + concave(curv) * 0.5;
    v *= 1.0 - p.edges_masking * convex(curv);
    var g = grunge(uv, f32(p.grunge_scale), p.seed_);
    if (p.use_triplanar != 0) {
        g = grunge_tri(load_position(c).rgb, wsn(load_world_space_normal(c).rgb), p.triplanar_blending_contrast,
                       f32(p.grunge_scale), p.seed_);
    }
    if (p.use_custom_grunge != 0) { g = load_grunge_input(c).r; }
    v = mix(v, v * g * 2.0, p.grunge_amount);
    store(c, g4(hscan(v, p.dirt_level, p.dirt_contrast) * load_mask(c).r));
}
