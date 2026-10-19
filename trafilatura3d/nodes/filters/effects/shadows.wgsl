// One pass: the largest share of samples that fall on the shape along a ray (times p.gain), over a cone that opens towards the light (angle p.theta,
// radians, in uv) from the pixel: RAYS rays across +- p.cone radians, Samples steps along each up to p.reach (uv).
// Only off the shape; white with the shadow in black.
const RAYS = 7;
const PEAK = 0.5;  // fitted on the output

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let here = clamp(load_input(c).r, 0.0, 1.0);
    let steps = clamp(p.samples, 1, 16);
    var darkest = 0.0;
    var total = 0.0;
    for (var r = 0; r < RAYS; r++) {
        let a = p.theta + p.cone * (2.0 * (f32(r) + 0.5) / f32(RAYS) - 1.0);
        let dir = vec2f(cos(a), sin(a));
        var hit = 0.0;
        for (var s = 0; s < steps; s++) {
            hit += sample_input(uv + dir * p.reach * (f32(s) + 0.5) / f32(steps)).r;
        }
        darkest = max(darkest, hit / f32(steps));
        total += hit / f32(steps);
    }
    // Between the cone's mean and its darkest direction (PEAK), scaled by Samples' gain.
    let shadow = min(mix(total / f32(RAYS), darkest, PEAK) * p.gain, 1.0);
    store(c, fx_gray(1.0 - clamp(shadow - here, 0.0, 1.0)));
}
