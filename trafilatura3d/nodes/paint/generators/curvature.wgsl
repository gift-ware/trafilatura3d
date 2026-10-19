fn effect(c: vec2i) -> vec4f {
    if (p.stage_ == 1) {
        var x = load_curv_(c).r;  // the bands weighted
        if (p.use_micro_details != 0) {
            x += micro_curvature(c, p.micro_normal, p.micro_height, p.curvature_type, p.curvature_intensity,
                                 p.height_details_intensity) - 0.5;
        }
        x = g_invert(g_curvature_mode(clamp(x, 0.0, 1.0), p.mode), p.invert);
        var m = clamp(g_contrast(x, p.contrast) + p.brightness, 0.0, 1.0);
        if (p.use_texture != 0) {
            let t = tex_texture(c, p.texture_scale, p.texture_triplanar, p.texture_triplanar_contrast, 0,
                                p.texture_grayscale_conversion, p.texture_invert, p.texture_contrast, p.texture_brightness);
            m = g_blend(m, t.x, p.texture_blending_mode, p.texture_opacity * t.y);
        }
        return vec4f(vec3f(m), 1.0);
    }
    return global_final(c);
}
