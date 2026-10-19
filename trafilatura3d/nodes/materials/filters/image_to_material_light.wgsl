// Levels: input (low, mid, high), output (low, high); mid is where the input maps to the middle.
fn lv(x: f32, lin: vec3f, lout: vec2f) -> f32 {
    let t = sat((x - lin.x) / max(lin.z - lin.x, 1e-4));
    let m = clamp((lin.y - lin.x) / max(lin.z - lin.x, 1e-4), 0.01, 0.99);
    return mix(lout.x, lout.y, pow(t, log(0.5) / log(m)));
}

fn relief(uv: vec2f) -> f32 {
    let h = sample_h(uv).r;
    let s = sample_hs(uv).r;
    return mix(h, s, p.normal_soften) + (h - s) * p.normal_sharpen * 3.0;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e = 1.0 / vec2f(d);
    let src = load_basecolor(c).rgb;
    let metal = load_metallic(c).r;
    let h = sample_h(uv).r;
    let l = luma(src);
    // Normal from the relief.
    let k = p.normal_strength * f32(d.x) / 256.0;
    let n = n_from4(relief(uv - vec2f(e.x, 0.0)), relief(uv + vec2f(e.x, 0.0)), relief(uv - vec2f(0.0, e.y)),
                    relief(uv + vec2f(0.0, e.y)), k);
    // AO: depth under the blurred surroundings, looked up towards the light.
    let la = p.ambient_occlusion_light_angle * TAU;
    let off = vec2f(cos(la), -sin(la)) * p.ambient_occlusion_spread * 0.01;
    let occ = sat((sample_hb(uv + off).r - h) * (2.0 + p.ambient_occlusion_light_distance * 10.0));
    let ao = mix(p.ambient_occlusion_levels.x, p.ambient_occlusion_levels.y, 1.0 - occ);
    // Diffuse adjustments.
    let fine = l - sample_lb(uv).r;
    var col = src * (1.0 + fine * p.sharpen * 4.0 / max(l, 0.05));
    var hsv = rgb2hsv(sat3(col));
    hsv.x = fract(hsv.x + p.hue);
    hsv.y = sat(hsv.y * p.saturation * 2.0);
    col = hsv2rgb(hsv);
    let ck = select(1.0 + p.contrast, 1.0 / (1.0 - 0.99 * p.contrast), p.contrast >= 0.0);
    col = (col - 0.5) * ck + 0.5 + (p.brightness - 0.5);
    col *= mix(1.0, ao, p.ambient_occlusion_in_diffuse);
    col = sat3(col);
    // Specular / glossiness / roughness from the luminance.
    let ls = l + fine * p.specular_sharpen * 4.0;
    let s = lv(mix(0.5, ls, p.specular_diffuse_influence), p.specular_levels_in, p.specular_levels_out);
    var spec = mix(vec3f(s), src * s * 2.0, p.specular_saturation) * 0.44;
    spec = mix(spec, col, metal * p.metallic_specular_influence);
    var gl = lv(mix(0.5, l, p.specular_diffuse_influence), p.glossiness_levels_in, p.glossiness_levels_out);
    gl = mix(gl, 0.8, metal * p.metallic_glossiness_influence);
    var ro = lv(mix(0.5, 1.0 - l, p.specular_diffuse_influence), p.roughness_levels_in, p.roughness_levels_out);
    ro = mix(ro, 0.3, metal * p.metallic_roughness_influence);
    let dx = p.output_normal_format == 0;
    store_diffuse(c, out_diffuse(c, vec4f(col * (1.0 - metal), 1.0)));
    store_basecolor(c, out_basecolor(c, vec4f(col, 1.0)));
    store_normal(c, out_normal(c, vec4f(n_enc(n, dx), 1.0)));
    store_specular(c, out_specular(c, vec4f(spec, 1.0)));
    store_emissive(c, vec4f(0.0, 0.0, 0.0, 1.0));
    store_glossiness(c, out_glossiness(c, g4(gl)));
    store_roughness(c, out_roughness(c, g4(ro)));
    store_metallic(c, out_metallic(c, g4(metal)));
    store_specular_level(c, g4(0.5));
    store_ambient_occlusion(c, out_ambient_occlusion(c, g4(ao)));
    store_height(c, out_height(c, g4(h)));
    store_opacity(c, g4(1.0));
}
