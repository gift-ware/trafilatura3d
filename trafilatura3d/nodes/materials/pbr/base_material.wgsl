fn user_h(uv: vec2f) -> f32 { return p.height_position + (sample_height(uv).r - 0.5) * p.height_range; }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e = 1.0 / vec2f(d);
    // Grunge, tiled.
    var gr = fbm(fract(uv * f32(p.grunge_tiling)), 4, 6, 0.55, p.seed_);
    if (p.custom_grunge_input != 0) { gr = sample_grunge(fract(uv * f32(p.grunge_tiling))).r; }
    let gv = (gr - 0.5) * p.grunge_amount;
    let gg = (gr - 0.5) * p.glossiness_grunge_amount;
    // Solid values per workflow and preset.
    let spec_f0 = dielectric_f0_range(p.specular_presets);
    let preset_spec = lin2srgb(vec3f(mix(spec_f0.x, spec_f0.y, p.specular_range)));
    var bc = p.base_color.rgb;
    var metal = p.metallic;
    var spec = p.specular.rgb;
    var diff = p.diffuse_color.rgb;
    if (p.material_preset == 1) { metal = 0.0; spec = preset_spec; }
    if (p.material_preset >= 2) {
        bc = lin2srgb(metal_f0(p.material_preset - 2));
        metal = 1.0;
        spec = bc;
        diff = vec3f(0.0);
    }
    var rough = sat(p.roughness_value + gv);
    var gloss = sat(p.glossiness_value - gg);
    if (p.pbr_workflow == 0) {
        if (p.material_preset != 1 && p.material_preset < 2) { spec = mix(preset_spec, bc, metal); }
        diff = bc * (1.0 - metal);
        gloss = 1.0 - rough;
    } else {
        if (p.material_preset < 2) {
            metal = smoothstep(0.35, 0.6, luma(srgb2lin(spec)));
            bc = mix(diff, spec, metal);
        }
        rough = 1.0 - gloss;
    }
    // User-defined maps override (their switches).
    if (p.user_basecolor != 0) { bc = load_basecolor(c).rgb; }
    if (p.user_diffuse != 0) { diff = load_diffuse(c).rgb; }
    if (p.user_specular != 0) { spec = load_specular(c).rgb; }
    if (p.user_metallic != 0) { metal = load_metallic(c).r; }
    if (p.user_roughness != 0) { rough = sat(load_roughness(c).r + gv); }
    if (p.user_glossiness != 0) { gloss = sat(load_glossiness(c).r - gg); }
    var h = p.height_position;
    var n = vec3f(0.0, 0.0, 1.0);
    if (p.user_height != 0) {
        h = user_h(uv);
        let k = p.normal_from_height_intensity * f32(d.x) / 256.0;
        n = n_from4(user_h(uv - vec2f(e.x, 0.0)), user_h(uv + vec2f(e.x, 0.0)), user_h(uv - vec2f(0.0, e.y)),
                    user_h(uv + vec2f(0.0, e.y)), k);
    }
    if (p.user_normal != 0) { n = n_combine(n_dec(load_normal(c).rgb, false), n); }
    var ao = 1.0;
    if (p.user_ao != 0) { ao = load_ambient_occlusion(c).r; }
    var em = vec3f(0.0);
    if (EMISSIVE_CONNECTED) { em = load_emissive(c).rgb; }
    var op = 1.0;
    if (OPACITY_CONNECTED) { op = load_opacity(c).r; }
    store_diffuse(c, vec4f(diff, 1.0));
    store_basecolor(c, vec4f(bc, 1.0));
    store_normal(c, vec4f(n_enc(n, false), 1.0));
    store_specular(c, vec4f(spec, 1.0));
    store_emissive(c, vec4f(em, 1.0));
    store_glossiness(c, g4(gloss));
    store_roughness(c, g4(rough));
    store_metallic(c, g4(metal));
    store_specular_level(c, g4(0.5));
    store_ambient_occlusion(c, g4(ao));
    store_height(c, g4(h));
    store_opacity(c, g4(op));
}
