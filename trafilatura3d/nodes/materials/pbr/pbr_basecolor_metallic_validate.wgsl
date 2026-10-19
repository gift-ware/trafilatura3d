// 0 = good, 0.5 = borderline, 1 = bad -> green, yellow, red.
fn heat(bad: f32) -> vec3f {
    return select(mix(vec3f(1.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), (bad - 0.5) * 2.0),
                  mix(vec3f(0.0, 0.8, 0.0), vec3f(1.0, 1.0, 0.0), bad * 2.0), bad < 0.5);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let bc = load_basecolor(c).rgb;
    let m = load_metallic(c).r;
    let s = luma(bc) * 255.0;  // sRGB value
    let lo = select(50.0, 30.0, p.albedo_dark_range_threshold == 1);
    // Albedo: distance outside [lo, 240] in sRGB steps, 10 steps = borderline -> bad at 30.
    let out_a = max(lo - s, s - 240.0);
    var albedo_bad = sat(out_a / 30.0 + 0.5) * step(0.0, out_a);
    albedo_bad *= 1.0 - step(0.5, m);
    // Metal: reflectance (linear luminance) under the minimum, and non-binary metallic.
    let refl = luma(srgb2lin(bc));
    let rmin = select(0.7, 0.6, p.metal_reflectance_range == 1);
    var metal_bad = sat((rmin - refl) / 0.2 + 0.5) * step(refl, rmin) * step(0.5, m);
    let mid = 1.0 - abs(m * 2.0 - 1.0);
    metal_bad = max(metal_bad, select(0.0, 0.75, mid > 0.2));
    var bad = albedo_bad;
    if (p.validation_mode == 1) { bad = metal_bad; }
    if (p.validation_mode == 2) { bad = max(albedo_bad, metal_bad); }
    var col = heat(bad);
    if (p.overlay_map != 0) { col *= 0.3 + 0.7 * luma(bc); }
    store(c, vec4f(col, 1.0));
}
