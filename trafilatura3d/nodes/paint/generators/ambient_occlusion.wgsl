fn effect(c: vec2i) -> vec4f {
    if (p.stage_ == 0) {  // the AO layer before its blur
        var a = load_baked_ambient_occlusion(c).r;
        if (p.use_micro_details != 0) {
            a *= micro_occlusion(c, p.micro_normal, p.micro_height, p.curvature_type, p.curvature_intensity,
                                 p.height_details_intensity, p.ao_radius, p.ao_depth);
        }
        return vec4f(g_invert(a, p.invert), 0.0, 0.0, 1.0);
    }
    if (p.stage_ == 1) {
        var m = g_adjust(sample_blur_r_(uv_of(c)).r, p.balance, p.contrast, 0.0);
        if (p.use_texture != 0) {
            let t = tex_texture(c, p.texture_scale, p.texture_triplanar, p.texture_triplanar_contrast, 0,
                                p.texture_grayscale_conversion, p.texture_invert, p.texture_contrast, p.texture_brightness);
            m = g_blend(m, t.x, p.texture_blending_mode, p.texture_opacity * t.y);
        }
        return vec4f(vec3f(m), 1.0);
    }
    return global_final(c);
}
