// Ambient occlusion (Deprecated), ours: the Equalizer on three gaussian bands of the AO (`fine` and `coarse` are its
// blurs): the coarse band through a gamma (Equalizer x: positive darkens, as a Levels mid of (x + 1) / 2), the middle
// and fine bands scaled by 1 + p.gains * Equalizer y / z. At 0, 0, 0 the AO unchanged.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let ao = load_ao(c).r;
    let fine = sample_fine(uv).r;
    let coarse = sample_coarse(uv).r;
    let e = p.equalizer;
    let gamma = select(1.0 / (1.0 - 8.0 * e.x), 1.0 + 8.0 * e.x, e.x >= 0.0);
    let v = pow(clamp(coarse, 1e-6, 1.0), gamma) + (1.0 + p.gains.x * e.y) * (fine - coarse)
          + (1.0 + p.gains.y * e.z) * (ao - fine);
    let o = clamp(v, 0.0, 1.0);
    store(c, vec4f(o, o, o, 1.0));
}
