// Direction in the environment's frame: the world direction turned back by Vertical then Horizontal Rotation.
fn env_dir(d: vec3f) -> vec3f {
    let h = radians(p.horizontal_rotation); let v = radians(p.vertical_rotation);
    let a = vec3f(d.x, d.y * cos(v) + d.z * sin(v), -d.y * sin(v) + d.z * cos(v));
    return vec3f(a.x * cos(h) - a.z * sin(h), a.y, a.x * sin(h) + a.z * cos(h));
}
// The prefiltered reflection for a roughness: the environment itself at 0, the four levels from 0.4 to 1.
fn reflection(d: vec3f, r: f32) -> vec3f {
    let uv = bl_dir_ll(d);
    let v = clamp(uv.y, 0.5 / 32.0, 1.0 - 0.5 / 32.0);
    let f = clamp((r - 0.4) / 0.2, 0.0, 3.0);
    let lo = floor(f); let hi = min(lo + 1.0, 3.0);
    let level = mix(sample_prefiltered(vec2f(uv.x, (v + lo) / 4.0)).rgb, sample_prefiltered(vec2f(uv.x, (v + hi) / 4.0)).rgb, f - lo);
    return mix(sample_environment(uv).rgb, level, clamp(r / 0.4, 0.0, 1.0));
}
fn effect(c: vec2i) -> vec4f {
    let n = bl_normal(c, true, p.height_intensity);
    let d = env_dir(n);
    let bc = load_basecolor(c);
    let b = bl_lin(bc.rgb);
    let m = load_metallic(c).r;
    let r = load_roughness(c).r;
    let ao = load_baked_ambient_occlusion(c).r;
    let f0 = mix(vec3f(0.04), b, m);
    // Lagarde's specular occlusion with the view along the normal (n.v = 1).
    let so = clamp(pow(1.0 + ao, exp2(-16.0 * r - 1.0)) - 1.0 + ao, 0.0, 1.0);
    let diffuse = b * (1.0 - m) * sample_irradiance(bl_dir_ll(d)).rgb * mix(1.0, ao, p.ambient_occlusion_intensity);
    let specular = f0 * reflection(d, r) * mix(1.0, so, p.specular_occlusion_intensity);
    return vec4f(bl_out((diffuse + specular) * exp2(p.exposure)), bc.a);
}
