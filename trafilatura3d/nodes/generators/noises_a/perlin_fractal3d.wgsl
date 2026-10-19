// VARIANT 0: Perlin levels; 1: ridged levels |n| (dark creases where the noise crosses zero, on a textured
// ground). Levels Min Level .. Max Level (1: the Scale), each Lacunarity times finer, Roughness times the previous.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var u = unit_slice(c);
    if (POSITION_MAP_CONNECTED) { u = load_position_map(c).rgb; }
    let tiled = p.enable_tiling != 0;
    let f0 = freq3(p.scale, p.size, tiled);
    let warp = distort3(u * f0, u, f0, tiled, p.distortion_intensity, p.distortion_scale_multiplier,
                        p.seed_ + 50.0) - u * f0;
    var sum = 0.0;
    var w1 = 0.0;
    var w2 = 0.0;
    var amp = 1.0;
    let lo = clamp(min(p.min_level, p.max_level) - 1, 0, 16);
    let hi = clamp(max(p.min_level, p.max_level) - 1, lo, 16);
    for (var k = lo; k <= hi; k++) {
        let f = freq3(p.scale * pow(max(p.lacunarity, 1.0), f32(k)), p.size, tiled);
        let x = u * f + p.offset + warp * f / f0;
        let n = perlin3(x, per3(f, tiled), p.seed_ + f32(k) * 7.31);
        var v = n;
        if (VARIANT == 1) {
            v = abs(n);
        } else {
#ABS#
        }
        sum += amp * v;
        w1 += amp;
        w2 += amp * amp;
        amp *= p.roughness;
    }
    var n = 0.0;
    if (VARIANT == 1) {
        // The ridge average in [0, 1] mapped to [-1, 1] around the Baseline.
        n = 2.0 * sum / max(w1, 1e-6) - 1.0;
    } else {
        // Normalized by the root of the summed squared weights: levels add detail, not contrast loss.
        n = sum / sqrt(max(w2, 1e-6));
    }
    var v = clamp(p.baseline + 0.325 * n * contrast3(p.contrast) * p.global_opacity, 0.0, 1.0);
    if (p.invert != 0) { v = 1.0 - v; }
    store(c, vec4f(v, v, v, 1.0));
}
