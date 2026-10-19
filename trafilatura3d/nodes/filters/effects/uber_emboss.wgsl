// Oren-Nayar diffuse for a viewer straight above (V = +Z).
fn oren_nayar(n: vec3f, l: vec3f, rough: f32) -> f32 {
    let s2 = rough * rough;
    let a = 1.0 - 0.5 * s2 / (s2 + 0.33);
    let b = 0.45 * s2 / (s2 + 0.09);
    let v = vec3f(0.0, 0.0, 1.0);
    let nl = max(dot(n, l), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let ti = acos(clamp(nl, 0.0, 1.0));
    let tr = acos(clamp(nv, 0.0, 1.0));
    let lp = normalize(l - n * dot(n, l) + vec3f(1e-6, 0.0, 0.0));
    let vp = normalize(v - n * dot(n, v) + vec3f(0.0, 1e-6, 0.0));
    let cphi = max(dot(lp, vp), 0.0);
    return nl * (a + b * cphi * sin(max(ti, tr)) * tan(min(ti, tr)));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let base = load_color(c);
    let n = fx_height_normal(c);
    let l = fx_light(p.light_angle);
    let lit = 1.0 - fx_cast_shadow(fx_uv(c), l) * p.shadows_opacity;
    let diff = oren_nayar(n, l, p.diffuse_roughness) * p.light_intensity * lit;
    let hv = normalize(l + vec3f(0.0, 0.0, 1.0));
    let e = exp2(1.0 + p.specular_glossiness * 9.0);
    let spec = pow(max(dot(n, hv), 0.0), e) * p.specular_intensity * lit * select(0.0, 1.0, dot(n, l) > 0.0);
    let rgb = base.rgb * (p.ambient_color.rgb + p.diffuse_color.rgb * diff) + p.specular_color.rgb * spec;
    store(c, vec4f(rgb, base.a));
}
