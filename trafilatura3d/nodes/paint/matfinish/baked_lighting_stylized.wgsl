// Sky: zenith color above, horizon color at the horizon, ground color below (sRGB colors).
fn sky(d: vec3f) -> vec3f {
    let up = mix(bl_lin(p.horizon_color), bl_lin(p.sky_color), smoothstep(0.0, 1.0, d.y));
    let down = mix(bl_lin(p.horizon_color), bl_lin(p.ground_color), smoothstep(0.0, 0.5, -d.y));
    return select(down, up, d.y >= 0.0) * p.sky_intensity;
}
// A directional light: (diffuse, specular) lighting, normalized Blinn-Phong with the view along the normal.
struct Lit { d: vec3f, s: vec3f };
fn light(n: vec3f, l: vec3f, rgb: vec3f, f0: vec3f, e: f32) -> Lit {
    let ndl = max(dot(n, l), 0.0);
    let h = normalize(l + n);
    let ndh = max(dot(n, h), 0.0);
    let f = f0 + (1.0 - f0) * pow(1.0 - ndh, 5.0);
    return Lit(rgb * ndl, rgb * ndl * f * (e + 8.0) / (8.0 * PI) * pow(ndh, e));
}
fn effect(c: vec2i) -> vec4f {
    let n = bl_normal(c, p.normal_details_type == 1, p.height_to_normal_intensity);
    let bc = load_basecolor(c);
    var albedo: vec3f; var f0: vec3f; var rough: f32;
    if (p.input == 0) {
        let m = load_metallic(c).r;
        let b = bl_lin(bc.rgb);
        albedo = b * (1.0 - m);
        f0 = mix(vec3f(p.dielectric_reflectance), b, m);
        rough = load_roughness(c).r;
    } else {
        albedo = bl_lin(load_diffuse(c).rgb);
        f0 = bl_lin(load_specular(c).rgb);
        rough = 1.0 - load_glossiness(c).r;
    }
    if (p.output == 1) { albedo = vec3f(1.0); f0 = vec3f(p.dielectric_reflectance); }
    let a = max(rough * rough, 0.03);
    let e = 2.0 / (a * a) - 2.0;
    // Occlusion: ambient occlusion and cavities (concave curvature, below 0.5); edges: convex curvature.
    let ao = load_baked_ambient_occlusion(c).r;
    let curv = load_curvature(c).r;
    let cav = 1.0 - smoothstep(0.0, mix(0.1, 1.0, p.cavity_smoothness), clamp((0.5 - curv) * 2.0, 0.0, 1.0));
    let edge = smoothstep(0.0, mix(0.1, 1.0, p.edges_smoothness), clamp((curv - 0.5) * 2.0, 0.0, 1.0));
    let occ_d = mix(1.0, ao, p.diffuse_ao) * mix(1.0, cav, p.diffuse_cavity);
    let occ_s = mix(1.0, ao, p.specular_ao) * mix(1.0, cav, p.specular_cavity);
    let sun = light(n, bl_dir(p.sun_horizontal_angle, p.sun_vertical_angle), bl_lin(p.sun_color) * p.sun_intensity, f0, e);
    let l1 = light(n, bl_dir(p.light1_horizontal_angle, p.light1_vertical_angle), bl_lin(p.light1_color) * p.light1_intensity, f0, e);
    let l2 = light(n, bl_dir(p.light2_horizontal_angle, p.light2_vertical_angle), bl_lin(p.light2_color) * p.light2_intensity, f0, e);
    let amb = sky(n);
    let diffuse = (sun.d + l1.d + l2.d + amb) * occ_d;
    let specular = (sun.s + l1.s + l2.s + amb * f0) * occ_s;
    let col = (albedo * diffuse + specular) * (1.0 + edge * p.edges_intensity);
    let out = vec4f(bl_out(col), select(bc.a, 1.0, p.input == 1));
    store_diffuse(c, out);
    return out;
}
