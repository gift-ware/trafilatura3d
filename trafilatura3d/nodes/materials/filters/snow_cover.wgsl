@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let m = read_mat(c, false);
    let hb = sample_hb(uv).r;
    let rel = m.height - hb;                       // > 0 raised, < 0 lowered
    let n = fbm(uv, 12, 5, 0.55, p.seed_) - 0.5;
    let fresh = hscan(m.height * 0.5 + rel * 2.0 + m.normal.z * 0.3 + n * 0.25, p.fresh_snow, 0.75);
    let melted = hscan(-rel * 3.0 + (1.0 - m.height) * 0.4 + n * 0.2, p.melted_snow, 0.6) * (1.0 - fresh);
    // Flakes: fine relief from a high-frequency noise.
    let e = 1.0 / vec2f(d);
    let f0 = vnoise(uv, vec2i(96), p.seed_ + 2.0);
    let fl = vec3f(-(vnoise(uv + vec2f(e.x, 0.0), vec2i(96), p.seed_ + 2.0) - f0),
                   -(vnoise(uv + vec2f(0.0, e.y), vec2i(96), p.seed_ + 2.0) - f0), 0.0) * f32(d.x) * 0.04;
    var snow = layer(m, vec3f(0.85, 0.87, 0.9), 0.6, 0.0);
    snow.normal = normalize(n_lerp(m.normal, vec3f(0.0, 0.0, 1.0), p.smoothness) + fl * p.flakes_intensity);
    snow.height = mix(m.height, hb, p.smoothness) + p.buildup * 0.25 * fresh;
    snow.ao = mix(m.ao, 1.0, 0.7);
    snow.specular_level = 0.5;
    var slush = layer(m, m.basecolor * 0.55, 0.12, 0.0);
    slush.normal = n_lerp(m.normal, vec3f(0.0, 0.0, 1.0), 0.7);
    slush.height = mix(m.height, hb, 0.5);
    let allw = Blend(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let k = load_mask(c).r;
    var o = blend_mat(m, slush, allw, melted * k);
    o = blend_mat(o, snow, allw, fresh * k);
    write_mat(c, o, false);
}
