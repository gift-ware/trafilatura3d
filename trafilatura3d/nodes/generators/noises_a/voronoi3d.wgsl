// The Random color style's stream: the node's (s) or Random color seed.
fn color_seed(s: f32) -> f32 {
    if (p.random_color_seed_mode == 1) { return f32(p.random_color_seed) * 101.0 + (s - p.seed_); }
    return s;
}

// Value of a cell sample for the styles F1 .. Edge (Random color handled by the caller).
fn style_value(r: Cell3, style: i32, dscale: f32, rounded: f32, thick: f32, aa: f32) -> f32 {
    let f1 = r.f1 * dscale;
    let f2 = r.f2 * dscale;
    var v = f1;
    if (style == 1) { v = f2; }
    else if (style == 2) { v = f2 - f1; }
    else if (style == 3) { v = f1 * f2; }
    else if (style == 4) { v = f1 / max(f2, 1e-6); }
    else if (style == 5) {
        // Edge: where the two nearest points are equally far (in cells, not scaled).
        return 1.0 - smoothstep(thick * 0.5 - aa, thick * 0.5 + aa, r.f2 - r.f1);
    }
    v = clamp(v, 0.0, 1.0);
    // Rounded Curve: the slope around each point becomes a round (convex) dome.
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
    let f = freq3(p.scale, p.size, tiled);
    var x = u * f + p.offset;
    x = distort3(x, u, f, tiled, p.distortion_intensity, p.distortion_scale_multiplier, p.seed_ + 50.0);
    let r = voronoi3(x, per3(f, tiled), vec3f(p.disorder), p.distance_mode, p.minkowski_number, p.seed_);
    if (p.style == 6) {
        var col = h33(r.id, color_seed(p.seed_) + 9.0);
        if (p.invert != 0) { col = 1.0 - col; }
        store(c, vec4f(col, 1.0));
        return;
    }
    let aa = max(f.x, f.y) / p.size_.x;
    var v = style_value(r, p.style, p.distance_scale, p.rounded_curve, p.edge_thickness, aa);
    if (p.invert != 0) { v = 1.0 - v; }
    store(c, vec4f(v, v, v, 1.0));
}
