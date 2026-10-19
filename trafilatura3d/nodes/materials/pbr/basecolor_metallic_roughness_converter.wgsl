@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let bc = srgb2lin(load_basecolor(c).rgb);
    let m = load_metallic(c).r;
    let r = load_roughness(c).r;
    var f0 = 0.04;
    if (p.use_specularlevel_input != 0) { f0 = 0.08 * load_specular_level(c).r; }
    let t = p.target_model;
    let diffuse = bc * (1.0 - m);
    var spec = mix(vec3f(f0), bc, m);
    // Fresnel-by-IOR renderers take a white reflection color for dielectrics.
    if (t == 1 || t == 2 || t == 3 || t == 4) { spec = mix(vec3f(1.0), bc, m); }
    var rough = r;
    if (t == 2) { rough = pow(r, 0.7); }          // legacy Corona glossiness curve
    if (t == 5) { rough = sqrt(r); }              // AiStandard specular roughness
    store_diffuse(c, vec4f(lin2srgb(diffuse), 1.0));
    store_specular(c, vec4f(lin2srgb(spec), 1.0));
    store_glossiness(c, g4(1.0 - rough));
    store_roughness(c, g4(rough));
}
