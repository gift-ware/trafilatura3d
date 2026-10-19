// How far Grunge Amount moves the edges, and the fibre value that leaves them (fitted on the output's statistics:
// the fibres only add wear).
const GRUNGE_GAIN = 0.8;
const GRUNGE_CENTRE = 0.0;

// Woven fibres: two crossing sets of soft stripes with noise, at uv.
fn weave(uv: vec2f, seed: f32) -> f32 {
    let q = uv * 48.0;
    let a = 0.5 + 0.5 * sin(q.x * TAU + vnoise(uv, vec2i(16), seed) * 3.0);
    let b = 0.5 + 0.5 * sin(q.y * TAU + vnoise(uv, vec2i(16), seed + 1.0) * 3.0);
    let checker = select(a, b, (i32(floor(q.x)) + i32(floor(q.y))) % 2 == 0);
    return checker * 0.7 + fbm(uv, 24, 3, 0.6, seed + 2.0) * 0.5;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    var f = weave(uv, p.seed_);
    if (p.use_triplanar != 0) {
        let pos = load_position(c).rgb;
        let w = tri_weights(wsn(load_world_space_normal(c).rgb), p.triplanar_blending_contrast);
        f = w.x * weave(fract(pos.zy), p.seed_) + w.y * weave(fract(pos.xz), p.seed_) + w.z * weave(fract(pos.xy), p.seed_);
    }
    if (p.use_custom_grunge != 0) { f = load_grunge_input(c).r; }
    // The blurred edge mask with the fibres added; occlusion takes away (1 - its factor) rather than scaling.
    let edges = sample_blurred(uv).r + (f - GRUNGE_CENTRE) * p.grunge_amount * GRUNGE_GAIN;
    let occl = occlusion(load_ambient_occlusion(c).r, p.ambient_occlusion_masking);
    store(c, g4(wear(edges + occl - 1.0, p.wear_level, p.wear_contrast) * load_mask(c).r));
}
