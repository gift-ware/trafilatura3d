// Input value at position x (0-1) along the section.
fn profile(x: f32) -> f32 {
    var q = vec2f(x, p.cross_section_coordinate);
    if (p.section_axis == 1) { q = vec2f(p.cross_section_coordinate, x); }
    return sample_input(q).r * p.height_scale + p.height_offset;
}

fn segments() -> f32 { return f32(max(p.segment_amount, 1)); }

// The profile drawn with straight segments between Segment amount + 1 points.
fn polygon(x: f32) -> f32 {
    let n = segments();
    let i = clamp(floor(x * n), 0.0, n - 1.0);
    return mix(profile(i / n), profile((i + 1.0) / n), x * n - i);
}

fn seg_dist(pt: vec2f, a: vec2f, b: vec2f) -> f32 {
    let ab = b - a;
    let h = clamp(dot(pt - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(pt - a - ab * h);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    let x = uv.x;
    let y = 1.0 - uv.y;  // height grows upwards
    var o = p.background_color;
    if (p.drawing_style == 3) {
        // Line: distance to the polyline, checking the segments around x.
        let n = segments();
        let th = max(p.line_thickness, 0.5 / p.size_.x);
        let reach = i32(ceil(th * n)) + 1;
        let i0 = i32(floor(x * n));
        var best = 1e9;
        var best_i = 0.0;
        for (var k = -reach; k <= reach; k++) {
            let i = f32(i0 + k);
            if (i < 0.0 || i >= n) { continue; }
            let a = vec2f(i / n, profile(i / n));
            let b = vec2f((i + 1.0) / n, profile((i + 1.0) / n));
            let dd = seg_dist(vec2f(x, y), a, b);
            if (dd < best) { best = dd; best_i = i; }
        }
        let aa = 1.0 / p.size_.x;
        let solid = 1.0 - smoothstep(th - aa, th + aa, best);
        let soft = 1.0 - smoothstep(th * (1.0 - clamp(p.line_smoothness, 0.0, 1.0)) - aa, th + aa, best);
        if (p.line_style == 0) { o = mix(o, p.color, soft); }
        else if (p.line_style == 1) { o = mix(o, p.color, solid); }
        else if (p.line_style == 2) { o = select(0.0, (best_i + 1.0) / n, solid > 0.5); }
        else { o = select(0.0, x, solid > 0.5); }
    } else {
        var v = profile(x);
        if (p.smooth_polygonal == 1) { v = polygon(x); }
        let aa = 1.0 / p.size_.y;
        if (p.drawing_style == 2) {
            // Mirrored about the middle line: the shape spans 0.5 +- v / 2.
            let r = abs(y - 0.5) * 2.0;
            let inside = 1.0 - smoothstep(v - aa, v + aa, r);
            var g = clamp(1.0 - r / max(v, 1e-6), 0.0, 1.0);
            if (p.invert_gradient == 1) { g = 1.0 - g; }
            o = mix(o, g, inside);
        } else {
            let inside = 1.0 - smoothstep(v - aa, v + aa, y);
            var fill = p.color;
            if (p.drawing_style == 1) {
                var g = clamp(y / max(v, 1e-6), 0.0, 1.0);
                if (p.invert_gradient == 1) { g = 1.0 - g; }
                fill = g;
            }
            o = mix(o, fill, inside);
        }
    }
    if (p.show_helper == 1) {
        let hs = max(p.helper_scale, 1e-4);
        let local = (uv - p.helper_position * (1.0 - hs)) / hs;  // Position 0 = top left, 1 = bottom right
        if (all(local >= vec2f(0.0)) && all(local <= vec2f(1.0))) {
            var h = sample_input(local).r;
            let along = select(local.y, local.x, p.section_axis == 1);
            if (abs(along - p.cross_section_coordinate) < 1.0 / (p.size_.x * hs)) { h = 1.0 - step(0.5, h); }
            o = h;
        }
    }
    store(c, fx_gray(o));
}
