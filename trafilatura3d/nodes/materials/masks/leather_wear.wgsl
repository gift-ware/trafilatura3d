@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let cl = cells(uv, 40, 0.9, p.seed_);
    var grain = sat(1.0 - cl.w * 4.0) * 0.6 + fbm(uv, 20, 4, 0.6, p.seed_ + 1.0) * 0.6;
    if (p.use_custom_grunge != 0) { grain = load_grunge_input(c).r; }
    var v = convex(load_curvature(c).r) * p.curvature_weight * 1.4 + (grain - 0.4) * p.grunge_amount;
    v *= mix(1.0, load_ambient_occlusion(c).r, p.ambient_occlusion_masking);
    store(c, g4(hscan(v, p.wear_level, p.wear_contrast) * load_mask(c).r));
}
