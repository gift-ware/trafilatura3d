const STEPS = 24;

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    let dist = load_intensity_input(c).r * p.intensity / 256.0;
    let step = dist / f32(STEPS);
    let first = i32(floor(f32(STEPS) * (1.0 - clamp(p.trail_length, 0.0, 1.0))));
    let expo = exp2(-2.0 * p.trail_curve);
    var q = uv;
    var acc = 0.0;
    var wsum = 0.0;
    var mx = -1e30;
    var mn = 1e30;
    for (var i = 1; i <= STEPS; i++) {
        let a = p.warp_angle + sample_warp_angle_input(q).r * p.warp_angle_input_multiplier;
        q += fx_dir(a) * step;
        if (i < first || i == STEPS) { continue; }
        // Trail: f = 0 at the end of the path, 1 at the far end of the trail.
        let f = f32(STEPS - i) / f32(max(STEPS - first, 1));
        let fade = clamp(p.trail_fade * pow(f, expo), 0.0, 1.0);
        let v = sample_input(q).r;
        mx = max(mx, v * (1.0 - fade));
        mn = min(mn, mix(v, 1.0, fade));
        acc += v * (1.0 - fade);
        wsum += 1.0 - fade;
    }
    let end = sample_input(q).r;
    var o = end;
    if (first < STEPS) {
        if (p.trail_mode == 0) { o = min(end, mn); }
        else if (p.trail_mode == 1) { o = max(end, mx); }
        else { o = (acc + end) / (wsum + 1.0); }
    }
    store(c, fx_gray(o));
}
