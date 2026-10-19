@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let dx = p.normal_format == 0;
    let m = read_mat(c, dx);
    var fx = m;
    // The field driving the effects: height, or flatness of the normal.
    var b = m.height;
    if (p.effect_from_height_or_normal == 1) { b = 1.0 - m.normal.z * 0.5 - 0.25; }
    let flow = vec2f(cos(p.flow_direction * TAU), sin(p.flow_direction * TAU));
    // Water in the low areas.
    let depth = p.water_level - b;
    let wet = smoothstep(-0.01, 0.02, depth) * step(0.001, p.water_level);
    let ripple = fbm(fract(uv + flow * 0.05), 8, 4, 0.5, p.seed_) - 0.5;
    let refr = sample_basecolor(uv + vec2f(ripple) * p.refraction * 0.05).rgb;
    let view = p.reflection_angle * PI * 0.5;
    let refl = vec3f(0.6, 0.7, 0.8) * p.reflection * smoothstep(0.0, 1.0, 1.0 - abs(uv.y - p.reflection_distance)) * sin(view + 0.3);
    let water_rgb = mix(refr, p.water_color.rgb, p.water_color.a * sat(depth * 4.0 + 0.3)) + refl;
    let wn = normalize(vec3f(ripple * p.water_details, ripple * p.water_details * 0.7, 1.0));
    var water = layer(m, water_rgb, 0.03, 0.0);
    water.normal = wn;
    water.height = max(m.height, p.water_level);
    water.ao = 1.0;
    // Ice: the water frozen, whiter and rougher, cracked.
    let cr = cells(uv, 12, 1.0, p.seed_ + 3.0);
    let crack = smoothstep(0.04, 0.0, cr.w) * p.ice_details;
    var ice = layer(m, mix(vec3f(0.75, 0.85, 0.9), water_rgb, 0.3) * (1.0 - crack * 0.5), 0.25 + crack * 0.5, 0.0);
    ice.normal = normalize(vec3f(-crack * 0.5, crack * 0.3, 1.0));
    ice.height = water.height;
    water = Mat(mix(water.diffuse, ice.diffuse, p.ice), mix(water.basecolor, ice.basecolor, p.ice),
                n_lerp(water.normal, ice.normal, p.ice), mix(water.specular, ice.specular, p.ice), water.emissive,
                mix(water.glossiness, ice.glossiness, p.ice), mix(water.roughness, ice.roughness, p.ice), 0.0,
                water.specular_level, water.ao, water.height, water.opacity);
    // Snow on high, flat areas.
    let snow_k = hscan(b * 0.6 + m.normal.z * 0.4 + (fbm(uv, 16, 4, 0.5, p.seed_ + 5.0) - 0.5) * 0.2, p.snow, 0.7);
    var snow = layer(m, vec3f(0.9, 0.92, 0.95), 0.6, 0.0);
    snow.height = m.height + snow_k * 0.05;
    snow.normal = n_lerp(m.normal, vec3f(0.0, 0.0, 1.0), 0.6);
    // Moss in the low areas, patchy.
    let mn = fbm(uv, 4 * p.moss_scale, 5, 0.6, p.seed_ + 7.0);
    let moss_k = hscan(mn * 0.7 + (1.0 - b) * 0.5, p.moss, 0.6);
    var moss = layer(m, p.moss_color.rgb * (0.7 + mn * 0.6), 0.9, 0.0);
    moss.height = m.height + moss_k * 0.03;
    // Stack: moss, water, snow.
    fx = Mat(mix(fx.diffuse, moss.diffuse, moss_k), mix(fx.basecolor, moss.basecolor, moss_k), fx.normal,
             mix(fx.specular, moss.specular, moss_k), fx.emissive, mix(fx.glossiness, moss.glossiness, moss_k),
             mix(fx.roughness, moss.roughness, moss_k), mix(fx.metallic, 0.0, moss_k), fx.specular_level,
             fx.ao, mix(fx.height, moss.height, moss_k), fx.opacity);
    fx = blend_mat(fx, water, Blend(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), wet);
    fx = blend_mat(fx, snow, Blend(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), snow_k);
    // Faked light on the colors.
    let la = p.light_angle * TAU;
    let ndl = dot(fx.normal, normalize(vec3f(cos(la), sin(la), 0.8)));
    let lit = mix(1.0, sat(0.4 + ndl), p.light_intensity);
    fx.diffuse *= lit;
    fx.basecolor *= lit;
    var k = 1.0;
    if (p.mask != 0) { k = load_mask(c).r; }
    let w = Blend(p.diffuse_intensity, p.base_color_intensity, p.normal_intensity, p.specular_intensity,
                  p.glossiness_intensity, p.roughness_intensity, 1.0, p.ambient_occlusion_intensity, p.height_intensity);
    write_mat(c, blend_mat(m, fx, w, k), dx);
}
