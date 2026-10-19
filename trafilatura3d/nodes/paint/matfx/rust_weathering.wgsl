// Rust mask with drips running down from it, then the rust material blended into each channel.
fn up_dir(uv: vec2f) -> vec2f {
    if (!POSITION_CONNECTED) { return vec2f(0.0, -1.0); }
    let o = 2.0 / p.size_;
    let y0 = sample_position(uv).g;
    let g = vec2f(sample_position(uv + vec2f(o.x, 0.0)).g - y0, sample_position(uv + vec2f(0.0, o.y)).g - y0);
    if (length(g) < 1e-6) { return vec2f(0.0, -1.0); }
    return normalize(g);
}

fn rust_at(uv: vec2f, up: vec2f) -> f32 {
    let r = sample_rust(uv).r;
    let n = p.drips_samples_amount;
    var drip = 0.0;
    for (var k = 1; k <= n; k++) {
        let t = f32(k) / f32(max(n, 1));
        drip = max(drip, sample_rust(uv + up * t * p.drips_intensity * 0.25).r * (1.0 - t));
    }
    let side = vec2f(-up.y, up.x);
    let streak = mx_noise(vec2f(fract(dot(uv, side) * 1.0), fract(dot(uv, up) * 1.0)), vec2i(96, 6), p.seed_ + 4.0);
    drip *= smoothstep(0.75 - p.drips_smoothness * 0.5, 0.85, streak);
    return max(r, drip * 0.85);
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let o = 1.0 / p.size_;
    let up = up_dir(uv);
    let r = rust_at(uv, up);
    let rn = mx_fbm(uv, 24, 4, 0.6, p.seed_ + 2.0);
    let rcol = mix(vec3f(0.22, 0.09, 0.04), vec3f(0.62, 0.30, 0.11), mx_contrast(rn, 0.4));
    let bc = load_basecolor(c);
    store_basecolor(c, vec4f(mix(bc.rgb, rcol, r * p.base_color_intensity), bc.a));
    let ro = load_roughness(c);
    store_roughness(c, vec4f(vec3f(mix(ro.r, 0.8 + rn * 0.15, r * p.roughness_intensity)), ro.a));
    let me = load_metallic(c);
    store_metallic(c, vec4f(vec3f(mix(me.r, 0.0, r * p.metallic_intensity)), me.a));
    let crust = r + rn * 0.3 * r;
    let cx = rust_at(uv + vec2f(o.x, 0.0), up);
    let cy = rust_at(uv - vec2f(0.0, o.y), up);
    let rnx = mx_fbm(uv + vec2f(o.x, 0.0), 24, 4, 0.6, p.seed_ + 2.0);
    let rny = mx_fbm(uv - vec2f(0.0, o.y), 24, 4, 0.6, p.seed_ + 2.0);
    let slope = mx_nslope(cx + rnx * 0.3 * cx - crust, cy + rny * 0.3 * cy - crust, 3.0 * mx_px() * p.normal_intensity);
    let n = load_normal(c);
    store_normal(c, vec4f(mx_nenc(mx_ndetail(mx_ndec(n.rgb), slope)), n.a));
    let h = load_height(c);
    store_height(c, vec4f(vec3f(h.r + crust * 0.03 * p.height_intensity), h.a));
    let ao = load_ambient_occlusion(c);
    store_ambient_occlusion(c, vec4f(vec3f(ao.r * mix(1.0, 0.55 + 0.45 * rn, r * p.ambient_occlusion_intensity)), ao.a));
}
