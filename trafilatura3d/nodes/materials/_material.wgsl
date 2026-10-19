// ---- materials/_material.wgsl: a full material at a pixel, for single-material nodes whose ports
// are the plain channel names (inputs and outputs) ----

struct Mat {
    diffuse: vec3f,
    basecolor: vec3f,
    normal: vec3f,          // decoded tangent-space unit vector (+Y up)
    specular: vec3f,
    emissive: vec3f,
    glossiness: f32,
    roughness: f32,
    metallic: f32,
    specular_level: f32,
    ao: f32,
    height: f32,
    opacity: f32,
}

// Blending strengths of an effect per channel (the "Blending" group).
struct Blend {
    diffuse: f32,
    basecolor: f32,
    normal: f32,
    specular: f32,
    glossiness: f32,
    roughness: f32,
    metallic: f32,
    ao: f32,
    height: f32,
}

fn read_mat(c: vec2i, directx: bool) -> Mat {
    return Mat(load_diffuse(c).rgb, load_basecolor(c).rgb, n_dec(load_normal(c).rgb, directx), load_specular(c).rgb,
               load_emissive(c).rgb, load_glossiness(c).r, load_roughness(c).r, load_metallic(c).r,
               load_specular_level(c).r, load_ambient_occlusion(c).r, load_height(c).r, load_opacity(c).r);
}

fn sample_mat(uv: vec2f, directx: bool) -> Mat {
    return Mat(sample_diffuse(uv).rgb, sample_basecolor(uv).rgb, n_dec(sample_normal(uv).rgb, directx),
               sample_specular(uv).rgb, sample_emissive(uv).rgb, sample_glossiness(uv).r, sample_roughness(uv).r,
               sample_metallic(uv).r, sample_specular_level(uv).r, sample_ambient_occlusion(uv).r, sample_height(uv).r,
               sample_opacity(uv).r);
}

fn write_mat(c: vec2i, m: Mat, directx: bool) {
    store_diffuse(c, out_diffuse(c, vec4f(m.diffuse, 1.0)));
    store_basecolor(c, out_basecolor(c, vec4f(m.basecolor, 1.0)));
    store_normal(c, out_normal(c, vec4f(n_enc(m.normal, directx), 1.0)));
    store_specular(c, out_specular(c, vec4f(m.specular, 1.0)));
    store_emissive(c, out_emissive(c, vec4f(m.emissive, 1.0)));
    store_glossiness(c, out_glossiness(c, g4(m.glossiness)));
    store_roughness(c, out_roughness(c, g4(m.roughness)));
    store_metallic(c, out_metallic(c, g4(m.metallic)));
    store_specular_level(c, out_specular_level(c, g4(m.specular_level)));
    store_ambient_occlusion(c, out_ambient_occlusion(c, g4(m.ao)));
    store_height(c, out_height(c, g4(m.height)));
    store_opacity(c, out_opacity(c, g4(m.opacity)));
}

// The effect `fx` laid over `m` with per-channel strengths times a mask.
fn blend_mat(m: Mat, fx: Mat, w: Blend, k: f32) -> Mat {
    var o = m;
    o.diffuse = mix(m.diffuse, fx.diffuse, w.diffuse * k);
    o.basecolor = mix(m.basecolor, fx.basecolor, w.basecolor * k);
    o.normal = n_lerp(m.normal, fx.normal, w.normal * k);
    o.specular = mix(m.specular, fx.specular, w.specular * k);
    o.glossiness = mix(m.glossiness, fx.glossiness, w.glossiness * k);
    o.roughness = mix(m.roughness, fx.roughness, w.roughness * k);
    o.metallic = mix(m.metallic, fx.metallic, w.metallic * k);
    o.ao = mix(m.ao, fx.ao, w.ao * k);
    o.height = mix(m.height, fx.height, w.height * k);
    o.emissive = mix(m.emissive, fx.emissive, k);
    o.opacity = mix(m.opacity, fx.opacity, k);
    o.specular_level = mix(m.specular_level, fx.specular_level, w.specular * k);
    return o;
}

// A surface layer (snow, rust, moss...) given by color, roughness, metallic: the other workflow's
// channels are derived (diffuse = albedo of dielectrics, specular = F0, glossiness = 1 - roughness).
fn layer(m: Mat, color: vec3f, rough: f32, metal: f32) -> Mat {
    var o = m;
    o.basecolor = color;
    o.diffuse = color * (1.0 - metal);
    o.specular = mix(vec3f(0.22), color, metal);
    o.roughness = rough;
    o.glossiness = 1.0 - rough;
    o.metallic = metal;
    return o;
}

