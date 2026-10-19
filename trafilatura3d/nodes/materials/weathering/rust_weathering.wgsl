fn rust_at(uv: vec2f) -> f32 {
    let place = (1.0 - sample_baked_ambient_occlusion(uv).r) * 0.5 + abs(sample_baked_curvature(uv).r - 0.5);
    let dmg = cells(uv, 4 + i32(p.varnish_damage_scale * 28.0), 1.0, p.seed_).x;
    let n = fbm(uv, 8, 5, 0.6, p.seed_ + 1.0);
    return hscan(place * 0.6 + (1.0 - dmg) * 0.15 + n * 0.5, p.rust_spreading, 1.0 - p.spreading_smoothness * 0.9);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e1 = 1.0 / vec2f(d);
    let dx = p.normal_format == 0;
    let m = read_mat(c, dx);
    var r = rust_at(uv);
    // Drips: rust found uphill (Position green) runs down to here.
    let h0 = sample_baked_position(uv).g;
    var g = vec2f(sample_baked_position(uv + vec2f(e1.x * 2.0, 0.0)).g - h0, sample_baked_position(uv + vec2f(0.0, e1.y * 2.0)).g - h0);
    var up = vec2f(0.0, -1.0);
    if (length(g) > 1e-5) { up = normalize(g); }
    let n = p.drips_samples_amount;
    var drip = 0.0;
    for (var k = 1; k <= n; k++) {
        let t = f32(k) / f32(max(n, 1));
        drip = max(drip, rust_at(uv + up * t * p.drips_intensity * 0.2) * (1.0 - t));
    }
    let side = vec2f(-up.y, up.x);
    drip *= smoothstep(0.4 - p.drips_smoothness * 0.3, 0.8, vnoise(vec2f(dot(uv, side), dot(uv, up) * 0.05), vec2i(64, 4), p.seed_ + 4.0));
    r = max(r, drip * 0.8);
    let rn = fbm(uv, 16, 4, 0.6, p.seed_ + 2.0);
    var fx = layer(m, mix(vec3f(0.38, 0.17, 0.07), vec3f(0.62, 0.32, 0.12), rn), 0.85, 0.0);
    let rx = rust_at(uv + vec2f(e1.x, 0.0));
    let ry = rust_at(uv - vec2f(0.0, e1.y));
    fx.normal = n_combine(m.normal, relief_normal(r + rn * 0.2, rx + rn * 0.2, ry + rn * 0.2, p.normal_intensity / 60.0 * f32(d.x) / 256.0));  // default 6
    fx.height = m.height + r * 0.02;
    var k = sat(r);
    if (p.mask != 0) { k *= load_mask(c).r; }
    let w = Blend(p.diffuse_intensity, p.base_color_intensity, min(p.normal_intensity, 1.0), p.specular_intensity,
                  p.glossiness_intensity, p.roughness_intensity, p.metallic_intensity, p.ambient_occlusion_intensity,
                  p.height_intensity);
    write_mat(c, blend_mat(m, fx, w, k), dx);
}
