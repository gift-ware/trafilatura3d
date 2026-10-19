// Brush strokes in four tone layers (background, shadows, midtones, highlights), each one stroke per grid cell;
// a pixel takes the top stroke (highest random priority) of each layer, layers painted in order.
// strokes: rgb = paint color, a = 1; aux: r = stroke body (thickness profile), g = bristles, b = stroke random.
struct Stroke { col: vec3f, alpha: f32, body: f32, bristle: f32, rnd: f32 };

fn tone(layer: i32, lum: f32) -> f32 {
    switch layer {
        case 0: { return 1.0; }
        case 1: { return smoothstep(0.42, 0.28, lum); }
        case 2: { return smoothstep(0.22, 0.36, lum) * smoothstep(0.8, 0.66, lum); }
        default: { return smoothstep(0.58, 0.72, lum); }
    }
}

fn bristles(x: f32, seed: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let a = hash21(vec2i(i32(i), 7), seed);
    let b = hash21(vec2i(i32(i) + 1, 7), seed);
    return mix(a, b, f * f * (3.0 - 2.0 * f));
}

fn layer_stroke(uv: vec2f, layer: i32, grid: i32, size: f32, details: f32, warp: f32) -> Stroke {
    var best = Stroke(vec3f(0.0), 0.0, 0.0, 0.5, 0.5);
    let n = max(i32(round(f32(grid) / max(p.global_scale_multiplier, 0.05))), 1);
    let seed = p.seed_ + f32(layer) * 101.0;
    let x = uv * f32(n);
    let wv = vec2f(mx_noise(uv, vec2i(n * 2), seed + 3.0), mx_noise(uv, vec2i(n * 2), seed + 4.0)) - 0.5;
    let xw = x + wv * warp * 0.8;
    let i = vec2i(floor(x));
    var prio_best = -1.0;
    for (var j = -2; j <= 2; j++) {
        for (var k = -2; k <= 2; k++) {
            let cell = i + vec2i(k, j);
            let q = wrap(cell, vec2i(n));
            let h = hash22(q, seed);
            let ctr = vec2f(cell) + 0.5 + (h - 0.5) * 0.9;
            let cuv = fract(ctr / f32(n));
            let src = sample_src(cuv);
            let present = tone(layer, mx_luma(src.rgb));
            if (hash21(q, seed + 1.0) >= present) { continue; }
            let prio = hash21(q, seed + 2.0);
            if (prio <= prio_best) { continue; }
            // Orientation: along the source's edges (structure tensor), else a slanted default.
            let t = sample_tensor(cuv);
            let base = mx_rot(vec2f(1.0, 0.0), 0.5 + (h.x - 0.5) * 0.6);
            let tr = t.x + t.y;
            var dir = base;
            if (tr > 1e-8) {
                let coh = sqrt((t.x - t.y) * (t.x - t.y) + 4.0 * t.z * t.z) / tr;
                let ang = 0.5 * atan2(2.0 * t.z, t.x - t.y) + PI * 0.5;
                var auto_dir = vec2f(cos(ang), sin(ang));
                if (dot(auto_dir, base) < 0.0) { auto_dir = -auto_dir; }
                dir = normalize(mix(base, auto_dir, mx_sat(p.automatic_orientation_multiplier * coh)) + vec2f(1e-5, 0.0));
            }
            let perp = vec2f(-dir.y, dir.x);
            let detail = mx_sat(t.w * 6.0);
            let r3 = hash22(q, seed + 5.0);
            let len = min(1.7 * size * p.size * (1.0 + p.scale_random * (r3.x - 0.5) * 1.6) * (1.0 - details * detail * 0.7), 4.0);
            let wid = len * 0.38 * (1.0 + p.size_random * (r3.y - 0.5) * 1.6);
            let rel = xw - ctr;
            let a = dot(rel, dir) / (len * 0.5);
            let taper = mix(1.0, 0.6, mx_sat((a + 1.0) * 0.5));
            let b = dot(rel, perp) / (wid * 0.5) / taper;
            let d = length(vec2f(max(abs(a) - 0.7, 0.0) / 0.3, b));
            let soft = 0.04 + 0.45 * (1.0 - p.strokes_hardness);
            var alpha = 1.0 - smoothstep(1.0 - soft, 1.0, d);
            if (alpha <= 0.02) { continue; }
            let br = bristles((b + 1.0) * 9.0 + h.y * 40.0, seed + 6.0);
            alpha *= 1.0 - smoothstep(0.35, 1.0, a) * (1.0 - br) * 0.9;  // dry brush at the stroke's end
            if (alpha <= 0.02) { continue; }
            let start = fract((ctr - dir * len * 0.35) / f32(n));
            let rc = hash22(q, seed + 8.0);
            let col = sample_src(start).rgb * (1.0 + (vec3f(rc.x, rc.y, hash21(q, seed + 9.0)) - 0.5) * p.random_color * 0.6);
            prio_best = prio;
            best = Stroke(mx_sat3(col), alpha, sqrt(mx_sat(1.0 - b * b)) * (1.0 - 0.3 * mx_sat(a)), br, prio);
        }
    }
    return best;
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    var col = sample_src(uv).rgb;
    var body = 0.0;
    var bristle = 0.5;
    var rnd = 0.5;
    for (var layer = 0; layer < 4; layer++) {
        var on = p.enabled_background; var grid = p.grid_subdivision_background; var size = p.strokes_size_background;
        var det = 0.0; var warp = p.warp_intensity_background;
        if (layer == 1) {
            on = p.enabled_shadows; grid = p.grid_subdivision_shadows; size = p.strokes_size_shadows;
            det = p.strokes_size_from_details_shadows; warp = p.warp_intensity_shadows;
        } else if (layer == 2) {
            on = p.enabled_midtones; grid = p.grid_subdivision_midtones; size = p.strokes_size_midtones;
            det = p.strokes_size_from_details_midtones; warp = p.warp_intensity_midtones;
        } else if (layer == 3) {
            on = p.enabled_highlights; grid = p.grid_subdivision_highlights; size = p.strokes_size_highlights;
            det = p.strokes_size_from_details_highlights; warp = p.warp_intensity_highlights;
        }
        if (on == 0) { continue; }
        let s = layer_stroke(uv, layer, grid, size, det, warp);
        col = mix(col, s.col, s.alpha);
        body = mix(body, s.body, s.alpha);
        bristle = mix(bristle, s.bristle, s.alpha);
        rnd = mix(rnd, s.rnd, s.alpha);
    }
    store_strokes(c, vec4f(col, 1.0));
    store_aux(c, vec4f(body, bristle, rnd, 1.0));
}
