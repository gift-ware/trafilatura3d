// A linear ramp per pixel: slope k from the contrast (an odds law, flat at 0, a step at 1), and its centre t
// slid by the position from just above white (position 0: all black) to just below black (position 1: all white).
const STEEPNESS = 20.0;  // k = STEEPNESS x (C / (1 - C))^SHARPNESS (fitted on the output)
const SHARPNESS = 2.8;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let pos = select(p.position, load_position_map(c).r + 2.0 * (p.position - 0.5), p.use_position_input != 0);
    let amount = p.contrast * select(p.contrast, load_contrast_map(c).r, p.use_contrast_input != 0);
    let cc = clamp(amount, 1e-4, 0.9999);
    let k = STEEPNESS * pow(cc / (1.0 - cc), SHARPNESS);
    let half = 0.5 / k;  // the ramp's half width in input values
    let t = mix(1.0 + half, -half, pos);
    store(c, gray4(clamp(0.5 + k * (load_input(c).r - t), 0.0, 1.0)));
}
