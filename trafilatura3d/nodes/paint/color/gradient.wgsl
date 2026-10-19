fn gradient_at(t: f32) -> vec3f {
    var col = array<vec3f, 8>(p.color_1, p.color_2, p.color_3, p.color_4, p.color_5, p.color_6, p.color_7, p.color_8);
    var pos = array<f32, 8>(p.color_1_position, p.color_2_position, p.color_3_position, p.color_4_position,
                            p.color_5_position, p.color_6_position, p.color_7_position, p.color_8_position);
    let n = clamp(p.color_quantity, 2, 8);
    var lo = -1;  // the key at or below t closest to it
    var hi = -1;  // the key at or above t closest to it
    for (var i = 0; i < n; i++) {
        if (pos[i] <= t && (lo < 0 || pos[i] > pos[lo])) { lo = i; }
        if (pos[i] >= t && (hi < 0 || pos[i] < pos[hi])) { hi = i; }
    }
    if (lo < 0) { return col[hi]; }
    if (hi < 0 || pos[hi] <= pos[lo]) { return col[lo]; }
    let f = (t - pos[lo]) / (pos[hi] - pos[lo]);
    if (p.blend_color_in_linear != 0) {
        return linear_to_srgb(mix(srgb_to_linear(col[lo]), srgb_to_linear(col[hi]), f));
    }
    return mix(col[lo], col[hi], f);
}

fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    if (p.display_gradient_debug != 0) { return vec4f(gradient_at(uv_of(c).x), 1.0); }
    let g = gray_of(clamp(v.rgb, vec3f(0.0), vec3f(1.0)), p.grayscale_mode);
    let t = balance_contrast(g, p.balance, p.contrast, p.invert);
    return vec4f(gradient_at(t), v.a);
}
