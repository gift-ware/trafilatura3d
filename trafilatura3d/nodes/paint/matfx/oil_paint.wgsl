// Strokes + fine details + sharpening, contour, canvas; the paint's relief into height / normal, roughness from
// strokes and bristles; Effect Intensity over the material.
fn canvas(uv: vec2f) -> f32 {
    let n = f32(p.fibers_count);
    let w = 0.5 + 0.25 * (sin(uv.x * n * 2.0 * PI) * select(1.0, -1.0, fract(uv.y * n * 0.5) > 0.5)
                         + sin(uv.y * n * 2.0 * PI) * select(1.0, -1.0, fract(uv.x * n * 0.5) > 0.5));
    return mx_sat(w * 0.8 + mx_noise(uv, vec2i(p.fibers_count), p.seed_ + 40.0) * 0.2);
}

fn relief(uv: vec2f) -> f32 {
    let a = sample_aux(uv);
    let k = p.normal_strokes_direction_influence;
    return a.r * (1.0 - 0.5 * k) + (a.g - 0.5) * k * 0.8 + canvas(uv) * p.texture_intensity * 0.3;
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = mx_px() / p.size_;
    let s = sample_strokes(uv).rgb;
    var ring = vec3f(0.0);
    var ring_src = vec3f(0.0);
    for (var i = 0; i < 4; i++) {
        let d = vec2f(select(-1.0, 1.0, (i & 1) == 1) * select(1.0, 0.0, i >= 2), select(-1.0, 1.0, (i & 1) == 1) * select(0.0, 1.0, i >= 2));
        ring += sample_strokes(uv + d * o).rgb;
        ring_src += sample_src(uv + d * o * 1.5).rgb;
    }
    ring *= 0.25;
    ring_src *= 0.25;
    var col = s + (s - ring) * p.sharpness * 2.0 + (sample_src(uv).rgb - ring_src) * p.fine_details * 1.5;
    // Contour: the strokes' edges, as thick as Thickness.
    let t = o * (0.5 + p.thickness * 3.0);
    let gx = mx_luma(sample_strokes(uv + vec2f(t.x, 0.0)).rgb) - mx_luma(sample_strokes(uv - vec2f(t.x, 0.0)).rgb);
    let gy = mx_luma(sample_strokes(uv + vec2f(0.0, t.y)).rgb) - mx_luma(sample_strokes(uv - vec2f(0.0, t.y)).rgb);
    col *= 1.0 - p.amount * smoothstep(0.03, 0.2, length(vec2f(gx, gy))) * 0.7;
    let cv = canvas(uv);
    col *= 1.0 - p.texture_intensity * 0.3 * (1.0 - cv);
    let e = p.effect_intensity;
    let bc = load_basecolor(c);
    store_basecolor(c, vec4f(mix(bc.rgb, mx_sat3(col), e), bc.a));
    let a = sample_aux(uv);
    let r = load_roughness(c);
    let rough = p.roughness + (a.b - 0.5) * p.roughness_variation + (a.g - 0.5) * p.roughness_details * 0.5;
    store_roughness(c, vec4f(vec3f(mix(r.r, mx_sat(rough), e)), r.a));
    let px = 1.0 / p.size_;
    let rv = relief(uv);
    let slope = mx_nslope(relief(uv + vec2f(px.x, 0.0)) - rv, relief(uv - vec2f(0.0, px.y)) - rv, 2.0 * mx_px() * p.normal_intensity * e);
    let n = load_normal(c);
    store_normal(c, vec4f(mx_nenc(mx_ndetail(mx_ndec(n.rgb), slope)), n.a));
    let h = load_height(c);
    store_height(c, vec4f(vec3f(h.r + rv * 0.02 * e), h.a));
}
