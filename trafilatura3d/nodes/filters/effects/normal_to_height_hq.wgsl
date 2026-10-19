// Final pass: auto levels (stats = max, -min of the relief) and Height Intensity.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var v = texel_relief(c).r;
    if (p.height_normalize == 1) {
        let s = textureLoad(src_stats, vec2i(0), 0);
        let lo = -s.y;
        v = (v - lo) / max(s.x - lo, 1e-6);
    }
    store(c, fx_gray(mix(0.5, v, clamp(p.height_intensity, 0.0, 1.0))));
}
