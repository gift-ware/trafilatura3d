// The Random color style's stream: the node's (s) or Random color seed.
fn color_seed(s: f32) -> f32 {
    if (p.random_color_seed_mode == 1) { return f32(p.random_color_seed) * 101.0 + (s - p.seed_); }
    return s;
}

fn style_value(r: Cell3, style: i32, dscale: f32, rounded: f32, thick: f32, aa: f32) -> f32 {
    let f1 = r.f1 * dscale;
    let f2 = r.f2 * dscale;
    var v = f1;
    if (style == 1) { v = f2; }
    else if (style == 2) { v = f2 - f1; }
    else if (style == 3) { v = f1 * f2; }
    else if (style == 4) { v = f1 / max(f2, 1e-6); }
    else if (style == 5) {
        return 1.0 - smoothstep(thick * 0.5 - aa, thick * 0.5 + aa, r.f2 - r.f1);
    }
    v = clamp(v, 0.0, 1.0);
    return mix(v, 1.0 - sqrt(max(1.0 - v * v, 0.0)), clamp(rounded, 0.0, 1.0));
}

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
    let lo = clamp(p.min_level, 0, 16);
    let hi = clamp(p.max_level, lo, 16);
    var acc = vec3f(0.0);
    if (p.blend_mode == 2) { acc = vec3f(1.0); }
    var wsum = 0.0;
    var amp = 1.0;
    for (var k = lo; k <= hi; k++) {
        let f = freq3(p.scale * pow(max(p.lacunarity, 1.0), f32(k)), p.size, tiled);
        let x = u * f + p.offset + warp * f / f0;
        let s = p.seed_ + f32(k) * 7.31;
        let r = voronoi3(x, per3(f, tiled), vec3f(p.disorder), p.distance_mode, p.minkowski_number, s);
        var v = vec3f(style_value(r, p.style, p.distance_scale, p.rounded_curve, p.edge_thickness,
                                  max(f.x, f.y) / p.size_.x));
        if (p.style == 6) { v = h33(r.id, color_seed(s) + 9.0); }
        if (p.blend_mode == 0) { acc += amp * v; wsum += amp; amp *= p.roughness; }
        else if (p.blend_mode == 1) { acc = max(acc, v); }
        else { acc = min(acc, v); }
    }
    if (p.blend_mode == 0) { acc /= max(wsum, 1e-6); }
    var o = clamp(acc * p.global_opacity, vec3f(0.0), vec3f(1.0));
    if (p.invert != 0) { o = 1.0 - o; }
    if (p.style == 6) { store(c, vec4f(o, 1.0)); } else { store(c, vec4f(o.x, o.x, o.x, 1.0)); }
}
