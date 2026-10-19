fn fetch(q: vec2f) -> vec4f {
    if (p.filtering_mode == 0) { return texel_input(vec2i(floor(q * vec2f(size_input())))); }
    return sample_input(q);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    var n = fx_dir(p.extension_angle);              // extension direction, across the section
    var origin = vec2f(0.5) + n * p.extension_position;
    var dist = max(p.extension_distance, 0.0);
    var t0 = -dist * 0.5;                            // band [t0, t1] along n from the section
    if (p.mode != 0) { t0 = 0.0; }
    if (p.mode == 2) {
        let v = p.end_position - p.start_position;
        dist = length(v);
        n = select(vec2f(1.0, 0.0), v / dist, dist > 1e-6);
        origin = p.start_position;
    }
    let t1 = t0 + dist;
    let t = dot(uv - origin, n);
    // Inside the band: the section's pixel; beyond it: the image moved by the band's width.
    var q = uv;
    if (t > t1) { q = uv - n * t1; }
    else if (t < t0) { q = uv - n * t0; }
    else { q = uv - n * t; }
    var o = fetch(q);
    if (INPUT_GRAY) {
        let k = clamp((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0);
        let off = mix(p.start_luminance_offset, p.end_luminance_offset, k);
        if (p.lum_offset_ignores_black_pixels == 0 || o.r > 0.0) { o = fx_gray(o.r + off); }
    }
    if (p.show_helper == 1) {
        let px = 1.5 / p.size_.x;
        let s = dot(uv - origin, vec2f(-n.y, n.x));
        var mark = select(0.0, 1.0, abs(t - t0) < px || abs(t - t1) < px);
        // Arrows along the extension, every 0.1 across the section, pointing away from it.
        let lane = abs(fract(s * 10.0 + 0.5) - 0.5) / 10.0;
        if (t > t0 && t < t1 && lane < px) { mark = 1.0; }
        // Arrow heads at the far edge (and at the near one too when the band grows both ways).
        let back1 = t1 - t;
        let back0 = select(1.0, t - t0, p.mode == 0);
        let head = min(back1, back0);
        if (t >= t0 && t <= t1 && head < 0.03 && lane < head * 0.6) { mark = 1.0; }
        let ink = select(vec4f(1.0, 0.45, 0.0, 1.0), vec4f(1.0), INPUT_GRAY);
        o = mix(o, ink, mark * 0.85);
    }
    store(c, o);
}
