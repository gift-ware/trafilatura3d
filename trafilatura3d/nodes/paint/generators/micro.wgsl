// Micro details: custom Micro Normal and Micro Height maps laid over the UVs (inputs `micro_normal`,
// `micro_height`). Differences are taken one texel of the micro map apart, so the result does not depend on the
// output size.

fn micro_n(uv: vec2f) -> vec2f { return g_normal(sample_micro_normal(uv)).xy; }

// Divergence of the micro normals' XY at a step of `k` micro texels (+Y up, rows grow downwards).
fn micro_div(uv: vec2f, k: f32, sobel: bool) -> f32 {
    let t = k / vec2f(size_micro_normal());
    let ex = vec2f(t.x, 0.0);
    let ey = vec2f(0.0, t.y);
    if (!sobel) {
        return (micro_n(uv + ex).x - micro_n(uv - ex).x + micro_n(uv - ey).y - micro_n(uv + ey).y) * 0.5;
    }
    var dx = 0.0;
    var dy = 0.0;
    for (var j = -1; j <= 1; j++) {
        let w = select(1.0, 2.0, j == 0);
        let o = f32(j);
        dx += w * (micro_n(uv + ex + ey * o).x - micro_n(uv - ex + ey * o).x);
        dy += w * (micro_n(uv - ey + ex * o).y - micro_n(uv + ey + ex * o).y);
    }
    return (dx + dy) / 8.0;
}

// Micro curvature around 0.5 (convex brighter): Curvature Type (0 Standard, 1 Sobel, 2 Smooth) times Curvature
// Intensity from the micro normal, plus the micro height's fine relief times Height Details Intensity.
fn micro_curvature(c: vec2i, use_normal: i32, use_height: i32, ctype: i32, cint: f32, hint: f32) -> f32 {
    let uv = uv_of(c);
    var x = 0.0;
    if (use_normal != 0 && MICRO_NORMAL_CONNECTED) {
        if (ctype == 2) {
            var acc = 0.0;
            for (var k = 0; k < 5; k++) { acc += micro_div(uv, exp2(f32(k)), false); }
            x += acc / 5.0 * 3.0;
        } else {
            x += micro_div(uv, 1.0, ctype == 1) * 2.0 * cint;
        }
    }
    if (use_height != 0 && MICRO_HEIGHT_CONNECTED) {
        let t = 2.0 / vec2f(size_micro_height());
        let h = sample_micro_height(uv).r;
        let ring = (sample_micro_height(uv + vec2f(t.x, 0.0)).r + sample_micro_height(uv - vec2f(t.x, 0.0)).r
                  + sample_micro_height(uv + vec2f(0.0, t.y)).r + sample_micro_height(uv - vec2f(0.0, t.y)).r) * 0.25;
        x += (h - ring) * 4.0 * hint;
    }
    return clamp(0.5 + x, 0.0, 1.0);
}

// Micro ambient occlusion of the micro height (horizon based, 8 directions): AO Radius = reach (up to 5% of the
// UV space), AO Depth = height of white. 1 = unoccluded.
fn micro_ao(c: vec2i, use_height: i32, radius: f32, depth: f32) -> f32 {
    if (use_height == 0 || !MICRO_HEIGHT_CONNECTED) { return 1.0; }
    let uv = uv_of(c);
    let r = max(radius * 0.05, 1.0 / f32(size_micro_height().x));
    let hs = depth * 0.05;
    let h0 = sample_micro_height(uv).r * hs;
    var occ = 0.0;
    for (var k = 0; k < 8; k++) {
        let a = (f32(k) + 0.5) / 8.0 * 2.0 * PI;
        let dir = vec2f(cos(a), sin(a));
        var sin_h = 0.0;
        for (var i = 1; i <= 6; i++) {
            let t = f32(i) / 6.0 * r;
            let dh = sample_micro_height(uv + dir * t).r * hs - h0;
            let s = dh / sqrt(dh * dh + t * t);
            if (s > sin_h) {
                occ += (s - sin_h) * (1.0 - t / (r * 1.0001));
                sin_h = s;
            }
        }
    }
    return clamp(1.0 - occ / 8.0, 0.0, 1.0);
}

// What micro details add to an ambient occlusion layer: the micro occlusion, darkened in the micro cavities.
fn micro_occlusion(c: vec2i, use_normal: i32, use_height: i32, ctype: i32, cint: f32, hint: f32, radius: f32,
                   depth: f32) -> f32 {
    let mc = micro_curvature(c, use_normal, use_height, ctype, cint, hint);
    return micro_ao(c, use_height, radius, depth) * min(1.0, mc * 2.0);
}
