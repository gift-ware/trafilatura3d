// Smooth 1D value noise in [-1, 1] along spline s.
fn scr_noise(u: f32, s: i32) -> f32 {
    let i = i32(floor(u));
    let f = u - floor(u);
    let a = srnd(vec2i(s, i), 90u);
    let b = srnd(vec2i(s, i + 1), 90u);
    return mix(a, b, f * f * (3.0 - 2.0 * f));
}

// Point of spline s's centreline at t in [0, 1], in its local frame (x along, y across).
fn scr_point(t: f32, s: i32, len: f32, amp: f32, freq: f32) -> vec2f {
    return vec2f((t - 0.5) * len, amp * scr_noise(t * freq + 17.0 * rnd1(s, 91u), s));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let fs = frame_size(p.non_square_expansion);
    let px = pixel_uv();
    let segs = max(p.max_segments_per_spline, 2);
    var best = 0.0;       // output value
    var depth = 0.0;      // groove depth of the deepest scratch (normal output)
    var slope = vec2f(0.0);
    for (var s = 0; s < max(p.spline_number, 1); s++) {
        let key = vec2i(s, 7);
        let ctr = (0.5 + (vec2f(rnd(key, 1u), rnd(key, 2u)) - 0.5) * p.spline_position_random) * fs;
        let ang = p.spline_rotation + p.spline_rotation_random * srnd(key, 3u) * 0.5;
        let len = max(p.spline_scale * (1.0 - p.spline_scale_random * rnd(key, 4u)), 1e-4);
        let amp = p.spline_distortion * (1.0 - p.spline_distortion_random * rnd(key, 5u)) * len * 0.15;
        let freq = 1.0 + p.spline_distortion_frequency * 20.0;
        let wk = p.spline_width * (1.0 - p.spline_width_random * rnd(key, 6u));
        let hw = select(wk * 0.75 / 2048.0, wk * 0.5 * px, p.set_spline_width_in_px == 1);
        if (hw <= 0.0) { continue; }
        let lum = 1.0 - p.luminance_random * rnd(key, 8u);
        let nint = p.normal_intensity * (1.0 - p.normal_intensity_random * rnd(key, 9u));
        let ex = 0.5 * len + hw + px;
        let ey = amp + hw + px;
        var d0 = uv - ctr;
        d0 -= round(d0 / fs) * fs;
        let r = rot(-ang);
        // The nearest periodic image and, per axis, the next one (long splines near a border).
        for (var k = 0; k < 4; k++) {
            var dd = d0;
            if ((k & 1) == 1) { dd.x -= sign(d0.x) * fs.x; }
            if ((k & 2) == 2) { dd.y -= sign(d0.y) * fs.y; }
            let l = r * dd;
            if (abs(l.x) > ex || abs(l.y) > ey) { continue; }
            let t = l.x / len + 0.5;
            let k0 = i32(floor(t * f32(segs)));
            var dm = 1e9;
            var tm = 0.0;
            var cm = vec2f(0.0);
            for (var kk = k0 - 2; kk <= k0 + 2; kk++) {
                if (kk < 0 || kk >= segs) { continue; }
                let t0 = f32(kk) / f32(segs);
                let t1 = f32(kk + 1) / f32(segs);
                let a = scr_point(t0, s, len, amp, freq);
                let b = scr_point(t1, s, len, amp, freq);
                let e = b - a;
                let h = clamp(dot(l - a, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
                let q = a + e * h;
                let dist = length(l - q);
                if (dist < dm) { dm = dist; tm = mix(t0, t1, h); cm = q; }
            }
            if (dm >= hw + px) { continue; }
            var fade = 1.0;
            let fl = max(p.fade_length, 1e-4);
            if (p.fade_mode == 1 || p.fade_mode == 3) { fade = min(fade, clamp(tm / fl, 0.0, 1.0)); }
            if (p.fade_mode == 2 || p.fade_mode == 3) { fade = min(fade, clamp((1.0 - tm) / fl, 0.0, 1.0)); }
            // Share of the pixel footprint (box filter across the line) that the scratch covers: a line
            // thinner than a pixel stays thin and faint instead of widening to a pixel.
            let cov = clamp((min(dm + 0.5 * px, hw) - max(dm - 0.5 * px, -hw)) / px, 0.0, 1.0);
            let prof = mix(1.0, sqrt(max(1.0 - (dm / hw) * (dm / hw), 0.0)), clamp(hw / px - 1.0, 0.0, 1.0));
            best = max(best, lum * fade * cov * prof);
            let dep = nint * fade * cov * prof;
            if (dep > depth) {
                depth = dep;
                // Groove: height falls towards the centreline, the slope points outwards.
                let dir = (l - cm) / max(dm, 1e-9);
                slope = transpose(r) * dir * (2.0 * nint * fade * cov * min(dm / hw, 1.0));
            }
        }
    }
    store_output(c, vec4f(best, best, best, 1.0));
    let ny = select(-slope.y, slope.y, p.normal_format == 1);
    let nrm = normalize(vec3f(-slope.x, ny, 1.0));
    store_normal(c, vec4f(nrm * 0.5 + 0.5, 1.0));
}
