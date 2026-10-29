// One pass: how much the heights along a ray stand above the pixel's (weighted mean of the steps, the far ones
// weighing less), over a cone that opens towards the light (angle p.theta, radians, in uv) from the pixel: RAYS rays
// across +- p.cone radians, Samples steps along each up to p.reach (uv), interleaved from ray to ray. Between the
// cone's mean and its darkest ray (p.peak), times p.gain, less BIAS; white with the shadow in black.
const RAYS = 7;
const FALLOFF = 0.4;  // the last step weighs 1 - FALLOFF of the first (fitted on the output)
const BIAS = 0.013;  // the faintest rise that shades (fitted on the output)

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let here = clamp(load_input(c).r, 0.0, 1.0);
    let steps = clamp(p.samples, 1, 16);
    let dt = 1.0 / f32(steps);
    let dw = -FALLOFF * dt;  // the weight's change per step
    let w0 = 1.0 + 0.5 * dw;  // the first step's weight
    let weights = f32(steps) * (1.0 - 0.5 * FALLOFF);  // the steps' weights add up to this
    var darkest = -1.0;
    var total = 0.0;
    for (var r = 0; r < RAYS; r++) {
        let a = p.theta + p.cone * (2.0 * (f32(r) + 0.5) / f32(RAYS) - 1.0);
        let step = vec2f(cos(a), sin(a)) * p.reach * dt;
        // Each ray starts a different fraction of a step out: the rays' steps interleave instead of repeating (with
        // the cone closed they would stack, and the shadow of a step would come in bands).
        let shift = (f32(r) + 0.5) / f32(RAYS) - 0.5;
        var at = uv + (0.5 + shift) * step;
        var w = w0 + shift * dw;
        var sum = 0.0;
        for (var s = 0; s < steps; s++) {
            sum += w * sample_input(at).r;
            at += step;
            w += dw;
        }
        let rise = sum / weights - here;
        darkest = max(darkest, rise);
        total += rise;
    }
    let shadow = mix(total / f32(RAYS), darkest, p.peak) * p.gain - BIAS;
    store(c, fx_gray(1.0 - clamp(shadow, 0.0, 1.0)));
}
