
fn vdist(d: vec3f) -> f32 {
    let a = abs(d);
    if (p.distance_mode == 1) { return a.x + a.y + a.z; }
    if (p.distance_mode == 2) { return max(a.x, max(a.y, a.z)); }
    if (p.distance_mode == 3) {
        let k = max(p.minkowski_number, 0.05);
        return pow(pow(a.x, k) + pow(a.y, k) + pow(a.z, k), 1.0 / k);
    }
    return length(d);
}

// F1, F2 and the random value of the nearest cell, on a lattice of period `per` (cells).
fn voronoi3(x: vec3f, per: vec3i, s: f32, cs: f32) -> vec3f {
    let i = vec3i(floor(x));
    var f1 = 9.0;
    var f2 = 9.0;
    var rnd = 0.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var xx = -1; xx <= 1; xx++) {
                let q = i + vec3i(xx, y, z);
                let w = wrap3(q, per);
                let h = vec3f(hash31(w, s), hash31(w, s + 1.7), hash31(w, s + 3.1));
                let pt = vec3f(q) + 0.5 + (h - 0.5) * clamp(p.disorder, vec3f(0.0), vec3f(1.0));
                let d = vdist(x - pt);
                if (d < f1) { f2 = f1; f1 = d; rnd = hash31(w, cs); }
                else if (d < f2) { f2 = d; }
            }
        }
    }
    return vec3f(f1, f2, rnd);
}

// One Voronoi sample: the style value t from the distances divided by Distance Scale and clamped to a cell's
// diagonal, sqrt(2) (Edge: 1 off the edges, F2 - F1 above Edge Thickness), drawn as the superellipse
// (1 - t^n)^(1/n), n = 1 + Rounded Curve^2.75 (1: 1 - t, 2: a quarter circle; fitted on the output); beyond t = 1
// it goes on below 0 as (2 - n)(1 - t), not clamped (the levels of the fractal add it).
fn vstyle(r: vec3f) -> f32 {
    let k = select(1.0, 1.0 / p.distance_scale, p.distance_scale != 0.0);
    let f1 = min(r.x * k, 1.41421356);
    let f2 = min(r.y * k, 1.41421356);
    var t = f1;
    switch (p.style) {
        case 1: { t = f2; }
        case 2: { t = abs(f2 - f1); }
        case 3: { t = f1 * f2; }
        case 4: { t = f1 / max(f2, 1e-5); }
        case 5: { t = select(0.0, 1.0, f2 - f1 > p.edge_thickness); }
        case 6: { t = r.z; }
        default: {}
    }
    t = max(t, 0.0);
    let n = 1.0 + pow(clamp(p.rounded_curve, 0.0, 1.0), 2.75);
    if (t >= 1.0) { return (2.0 - n) * (1.0 - t); }
    return pow(1.0 - pow(t, n), 1.0 / n);
}

fn color_seed() -> f32 {
    return select(p.seed_, f32(p.random_color_seed), p.random_color_seed_mode == 1) + 101.0;
}

// The sample point of one level: uv warped by Distortion, in lattice units of `cells` per tile.
fn vpoint(uv: vec2f, cells: vec3f) -> vec3f {
    var q = uv;
    if (p.distortion_intensity != 0.0) {
        let wp = max(i32(round(p.scale * max(p.distortion_scale_multiplier, 0.0))), 1);
        q += p.distortion_intensity * 0.1 * warp_vec(uv, wp, 3, p.seed_ + 77.0);
    }
    return (vec3f(q, 0.0) + p.offset) * cells + vec3f(0.0, 0.0, 0.25);  // a slice a quarter into the cells
}
