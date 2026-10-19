// A per-tile cell key from the data (tile index and random), for per-tile random values.
fn ssb_key(d1: vec4f, d2: vec4f) -> vec2i {
    return vec2i(i32(round(d2.a * 1048576.0)), i32(round(d1.a * 1048576.0)));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let d1 = load_splatter_data_1(c);
    let d2 = load_splatter_data_2(c);
    let bg = load_background_color(c);
    let ci = load_color_input(c);
    var o = bg;
    let pid = i32(round(d1.b));
    if (pid >= 1) {
        let key = ssb_key(d1, d2);
        let np = clamp(p.pattern_number, 1, 8);
        var k = wrapi(pid - 1, np);
        if (rnd(key, 1u) < p.random_pattern_assignment) { k = min(i32(rnd(key, 2u) * f32(np)), np - 1); }
        let uv = clamp(d1.rg, vec2f(0.0), vec2f(1.0));
        var col: vec4f;
        if (ss_connected(k)) {
            col = pat_sample(k, uv, 1, 0.0);
        } else {
            let s = shape(2, uv * 2.0 - 1.0, 0.0, 0.01);
            col = vec4f(s, s, s, 1.0);
        }
        if (!OUT_OUTPUT_GRAY && p.is_normal_map == 1) {
            // OpenGL normals: green flipped in and out, so the rotation follows the tile on screen.
            let fy = vec3f(1.0, select(1.0, -1.0, p.normal_format == 1), 1.0);
            let nv = (col.rgb * 2.0 - 1.0) * fy;
            let ang = d2.r + p.normal_angle_random * srnd(key, 3u) * 0.5;
            let xy = rot(-ang) * nv.xy;
            col = vec4f((normalize(vec3f(xy, nv.z)) * fy) * 0.5 + 0.5, col.a);
            col = vec4f(mix(col.rgb, ci.rgb, p.color_input_opacity), col.a);
        } else if (OUT_OUTPUT_GRAY) {
            let v = mix(col.r, ci.r, p.color_input_opacity) + p.luminance_adjustment + rnd(key, 4u) * p.luminance_random;
            col = vec4f(vec3f(clamp(v, 0.0, 1.0)), col.a);
        } else {
            let a = clamp(p.hsl_adjustment + vec3f(rnd(key, 4u), rnd(key, 5u), rnd(key, 6u)) * p.hsl_random,
                          vec3f(-1.0), vec3f(1.0));
            let h = rgb_to_hsl(clamp(mix(col.rgb, ci.rgb, p.color_input_opacity), vec3f(0.0), vec3f(1.0)));
            // Saturation and lightness towards 0 (negative) or 1 (positive).
            let sl = mix(mix(h.yz, vec2f(0.0), abs(min(a.yz, vec2f(0.0)))), vec2f(1.0), max(a.yz, vec2f(0.0)));
            col = vec4f(hsl_to_rgb(vec3f(fract(h.x + a.x), sl)), col.a);
        }
        let alpha = select(col.a, 1.0, OUT_OUTPUT_GRAY);
        o = vec4f(mix(bg.rgb, col.rgb, alpha), max(bg.a, alpha));
    }
    if (OUT_OUTPUT_GRAY) { store(c, vec4f(o.r, o.r, o.r, 1.0)); } else { store(c, o); }
}
