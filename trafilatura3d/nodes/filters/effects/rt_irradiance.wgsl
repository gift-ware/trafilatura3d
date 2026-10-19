fn ray_height(uv: vec2f) -> f32 { return sample_height(uv).r * p.height_scale; }

fn to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

fn to_srgb(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}

fn emission(uv: vec2f, r: f32) -> vec3f {
    var acc = vec3f(0.0);
    for (var k = 0; k < 4; k++) {
        let o = fx_rot(vec2f(r, 0.0), f32(k) * PI * 0.5 + 0.785);
        var e = sample_emissive(uv + o).rgb;
        if (p.emissive_color_space == 0) { e = to_linear(max(e, vec3f(0.0))); }
        acc += e;
    }
    return acc * 0.25 * p.emissive_intensity;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    let n = ray_normal(uv);
    let count = select(select(128, 64, p.quality == 1), 32, p.quality == 0);
    let r = hash22(c, p.seed_ + 21.0);
    let exposure = exp2(p.environment_exposure_ev);
    let reach = 0.25;
    var light = vec3f(0.0);
    var open = 0.0;
    for (var i = 0; i < count; i++) {
        let dir = ray_dir(i, count, r, n, 1.0, true);
        let t = ray_march(uv, dir, reach, 16, fract(r.y + f32(i) * 0.618));
        if (t < 0.0) {
            light += sample_environment(fx_latlong(dir, p.environment_rotation)).rgb * exposure;
            open += 1.0;
        } else {
            let hit = uv + vec2f(dir.x, -dir.y) * t;
            let foot = max(1.0 / p.size_.x, t * 0.25 * exp2(p.emissive_lod_bias * 2.0));
            light += emission(hit, foot);
            if (p.bounce == 1) { light += sample_previous(hit).rgb; }
        }
    }
    let albedo = to_linear(max(load_basecolor(c).rgb, vec3f(0.0))) * (1.0 - clamp(load_metallic(c).r, 0.0, 1.0));
    light *= albedo / f32(count);
    let alpha = select(1.0, open / f32(count), p.ibl_shadows_in_raw_irradiance_alpha == 1);
    store_raw_irradiance(c, vec4f(light, alpha));
    store_output(c, vec4f(to_srgb(emission(uv, 0.0) + light), 1.0));
}
