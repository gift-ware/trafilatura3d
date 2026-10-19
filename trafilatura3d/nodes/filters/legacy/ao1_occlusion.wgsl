// Ambient occlusion, ours: `blurred` (the height's gaussian, read at an offset towards the light) is the reference;
// how far the height lies under it darkens as exp(-(under / p.depth)^p.power); what rises above it (times p.relief)
// plus the facing 1 / (1 + (p.face x slope)^2) of the height (slope of its gaussian of about a pixel, in 256 px
// units) is the light it gets, plus p.light x the slope of `lit` facing the light (scale p.light_k); clamped, then
// the output levels p.range (low, high).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let h = load_height(c).r;
    let b = sample_blurred(uv + p.shift).r;
    let shade = exp(-pow(max(b - h, 0.0) / p.depth, p.power));
    // Slope of the height's gaussian (sigma 0.875 px): derivative-of-gaussian weights on 5 x 5 texels.
    var g = vec2f(0.0);
    var z = 0.0;
    for (var j = -2; j <= 2; j++) {
        for (var i = -2; i <= 2; i++) {
            let o = vec2f(f32(i), f32(j));
            let w = exp(-dot(o, o) * 0.6530612);  // 1 / (2 sigma^2)
            g += load_height(c + vec2i(i, j)).r * o * w;
            z += w;
        }
    }
    let k = p.face * f32(d.x) / 256.0;
    let slope = g / (z * 0.765625) * k;  // / sigma^2, per 256 px
    var light = p.relief * max(h - b, 0.0) + 1.0 / (1.0 + dot(slope, slope));
    let e = 1.0 / vec2f(d);
    let gl = vec2f(sample_lit(uv + vec2f(e.x, 0.0)).r - sample_lit(uv - vec2f(e.x, 0.0)).r,
                   sample_lit(uv + vec2f(0.0, e.y)).r - sample_lit(uv - vec2f(0.0, e.y)).r) * f32(d.x) / 512.0;
    light += p.light * clamp(-p.light_k * dot(gl, p.light_dir), 0.0, 1.0);
    let ao = shade * clamp(light, 0.0, 1.0);
    let v = clamp((ao - p.range.x) / max(p.range.y - p.range.x, 1e-4), 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
