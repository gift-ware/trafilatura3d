// How far Grunge Amount moves the edges, and the grunge value that leaves them (fitted on the output's statistics).
const GRUNGE_GAIN = 1.8;
const GRUNGE_CENTRE = 0.35;

fn metal_grunge(uv: vec2f, scale: f32, seed: f32) -> f32 {
    let s = scratches(uv, i32(scale) * 6, 1.2, 0.3, 1.5, 0.05, seed + 5.0);
    return max(grunge(uv, scale, seed), s);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let sc = f32(p.grunge_scale);
    var g = metal_grunge(uv, sc, p.seed_);
    if (p.use_triplanar != 0) {
        let pos = load_position(c).rgb;
        let w = tri_weights(wsn(load_world_space_normal(c).rgb), p.triplanar_blending_contrast);
        g = w.x * metal_grunge(fract(pos.zy), sc, p.seed_) + w.y * metal_grunge(fract(pos.xz), sc, p.seed_)
            + w.z * metal_grunge(fract(pos.xy), sc, p.seed_);
    }
    if (p.use_custom_grunge != 0) { g = load_grunge_input(c).r; }
    // The blurred edge mask with the grunge added, scaled down where occluded.
    let edges = sample_blurred(uv).r + (g - GRUNGE_CENTRE) * p.grunge_amount * GRUNGE_GAIN;
    let occl = occlusion(load_ambient_occlusion(c).r, p.ambient_occlusion_masking);
    store(c, g4(wear(edges * occl, p.wear_level, p.wear_contrast) * load_mask(c).r));
}
