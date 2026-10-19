// RTAO / Bent Normal: height in UV = the input x Height Scale (white = that
// fraction of the width) or x Physical Size Z / min(X, Y).
fn ray_height(uv: vec2f) -> f32 {
    var hs = p.height_scale;
    if (p.use_physical_size == 1) {
        let s = min(p.physical_size.x, p.physical_size.y);
        hs = select(p.physical_size.z / s, 0.0, s == 0.0);
    }
    return sample_input(uv).r * hs;
}

// Rays in a cone around the normal (cosine-distributed or uniform), unweighted: x = the share that
// escapes within Maximum Distance (1 = open), yzw = the sum of their directions (bent normal).
fn trace(c: vec2i) -> vec4f {
    let uv = fx_uv(c);
    let n = ray_normal(uv);
    let count = clamp(p.samples, 1, 2048);
    let cosine = p.distribution == 0;
    let r = hash22(c, p.seed_ + 3.0);
    var vis = 0.0;
    var bent = vec3f(0.0);
    for (var i = 0; i < count; i++) {
        let d = ray_dir(i, count, r, n, p.spread_angle, cosine);
        let hit = ray_march(uv, d, max(p.maximum_distance, 1e-4), 16, fract(r.x * 7.31 + f32(i) * 0.618));
        let open = select(1.0, 0.0, hit >= 0.0);
        vis += open;
        bent += d * open;
    }
    return vec4f(vis / f32(count), bent);
}
