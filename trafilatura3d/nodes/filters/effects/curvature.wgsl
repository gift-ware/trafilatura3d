// Grey per unit of divergence per pixel at Intensity 1 (fitted on the output: the same at every resolution).
const GAIN = 0.4;

fn nrm(c: vec2i) -> vec3f { return fx_decode_normal(load_input(c), p.normal_format == 0); }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // +Y is up, rows grow downwards.
    let div = (nrm(c + vec2i(1, 0)).x - nrm(c - vec2i(1, 0)).x + nrm(c - vec2i(0, 1)).y - nrm(c + vec2i(0, 1)).y) * 0.5;
    store(c, fx_gray(clamp(0.5 + div * GAIN * p.intensity, 0.0, 1.0)));
}
