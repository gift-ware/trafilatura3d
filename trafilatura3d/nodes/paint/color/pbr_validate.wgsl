fn effect(c: vec2i) -> vec4f {
    let bc = load_basecolor(c);
    let metal = load_metallic(c).r >= 0.5;
    // Brightness of the base color in sRGB 0-255: the linear luminance, encoded.
    let lum = linear_to_srgb(vec3f(dot(srgb_to_linear(clamp(bc.rgb, vec3f(0.0), vec3f(1.0))),
                                       vec3f(0.2126, 0.7152, 0.0722)))).x * 255.0;
    let albedo_min = select(30.0, 50.0, p.albedo_dark_range_threshold == 1);
    let metal_min = select(180.0, 155.0, p.metal_reflectance_range == 1);
    let checked = select(p.validation_mode != 1, p.validation_mode != 0, metal);
    let valid = select(lum >= albedo_min, lum >= metal_min, metal);
    let flag = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), valid);
    if (p.overlay_map != 0) {
        return vec4f(select(bc.rgb, mix(bc.rgb, flag, 0.5), checked), bc.a);
    }
    return vec4f(select(vec3f(0.0), flag, checked), bc.a);
}
