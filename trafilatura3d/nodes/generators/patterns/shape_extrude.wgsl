// The extrude profile: up and down by half the frame x Scale x Extrude Height / Depth (Custom Input:
// before the whole result is scaled, see se_cast).
fn se_zrange() -> vec2f {
    let k = select(p.scale, 1.0, p.extrude_shape == 2);
    return 0.5 * k * vec2f(-p.extrude_depth, p.extrude_height);
}

fn se_inside(q: vec3f) -> bool {
    let z = se_zrange();
    if (q.z < z.x || q.z > z.y) { return false; }
    let len = max(z.y - z.x, 1e-5);
    let u = (q.z - z.x) / len;
    var s = 1.0;
    if (p.profile_type == 1) { s = sample_profile_gradient(vec2f(u, 0.5)).r; }
    if (p.profile_type == 2 && sample_profile_mask(vec2f(u, 0.5)).r < 0.5) { return false; }
    // Bevel: the section shrinks near the top (and the bottom when mirrored).
    let bl = max(p.bevel_height * len * 0.5, 1e-5);
    let k = exp2(-2.0 * p.bevel_curve);
    let t_top = clamp(1.0 - (z.y - q.z) / bl, 0.0, 1.0);
    var f = pow(t_top, k);
    if (p.mirror_bevel == 1) { f = max(f, pow(clamp(1.0 - (q.z - z.x) / bl, 0.0, 1.0), k)); }
    s *= 1.0 - p.bevel_intensity * f;
    if (s <= 1e-4) { return false; }
    let xy = q.xy / s;
    let half = max(0.5 * p.extrude_shape_size * p.scale, vec2f(1e-5));
    switch p.extrude_shape {
        case 0: { return all(abs(xy) <= half); }
        case 1: { return length(xy / half) <= 1.0; }
        default: {
            let uv = vec2f(xy.x + 0.5, 0.5 - xy.y);
            if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return false; }
            return sample_extrude_shape_input(uv).r >= 0.5;
        }
    }
}

// Height of the visible surface at screen point s (world units), or -1e9 when the ray misses.
fn se_cast(s: vec2f) -> f32 {
    let custom = p.extrude_shape == 2;
    let k = select(1.0, max(p.scale, 1e-4), custom);  // Custom Input: Scale scales the whole result
    // Position: v =2 x position - 1 (y down), tilt |v| / 4 turns about v's direction.
    let v = 2.0 * p.position - 1.0;
    let az = select(0.0, atan2(-v.y, v.x) / TAU, length(v) > 1e-6);
    let m = rot3(vec3f(0.0, 0.0, az)) * rot3(vec3f(0.25 * length(v), 0.0, 0.0)) * rot3(vec3f(0.0, 0.0, -az));
    let mt = transpose(m);
    let o = mt * vec3f(s / k, 4.0);
    let dir = mt * vec3f(0.0, 0.0, -1.0);
    let z = se_zrange();
    let half = select(0.5 * p.extrude_shape_size * p.scale, vec2f(0.5), custom);
    // Bounding box of the volume (centred on its z range).
    let zc = 0.5 * (z.x + z.y);
    let t = ray_box(o - vec3f(0.0, 0.0, zc), dir, vec3f(max(half, vec2f(1e-5)), max(0.5 * (z.y - z.x), 1e-5)));
    if (t.x > t.y) { return -1e9; }
    let n = 192;
    var lo = t.x;
    var hi = t.y;
    var found = se_inside(o + dir * t.x);
    if (found) { hi = t.x; }
    for (var i = 1; i <= n && !found; i++) {
        let tt = mix(t.x, t.y, f32(i) / f32(n));
        if (se_inside(o + dir * tt)) { hi = tt; found = true; } else { lo = tt; }
    }
    if (!found) { return -1e9; }
    for (var i = 0; i < 8 && hi > lo; i++) {
        let tt = 0.5 * (lo + hi);
        if (se_inside(o + dir * tt)) { hi = tt; } else { lo = tt; }
    }
    return (4.0 - hi) * k;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // Downscale Multiplier: one ray at the centre of each 2^n x 2^n block.
    let b = 1 << u32(clamp(p.downscale_multiplier, 0, 2));
    let s = screen_point(c, 0) + vec2f(vec2i(b) * (c / vec2i(b)) - c) * vec2f(1.0, -1.0) / p.size_
        + vec2f(0.5 * f32(b - 1), -0.5 * f32(b - 1)) / p.size_;
    let h = se_cast(s);
    var v = 0.0;
    if (h > -1e8) { v = select(0.5 + 0.5 * h, h, p.output_range == 1); }
    store(c, vec4f(v, v, v, 1.0));
}
