fn effect(c: vec2i) -> vec4f {
    if (p.stage_ == 1) {
        let pos = clamp(load_position(c).rgb, vec3f(0.0), vec3f(1.0));
        var x = g_three(pos.zyx, vec3f(p.right_to_left, p.top_to_bottom, p.front_to_back),
                        vec3i(p.rl_invert, p.tb_invert, p.fb_invert),
                        vec3i(p.rl_blending_mode, p.tb_blending_mode, p.fb_blending_mode));
        var m = g_adjust(g_invert(x, p.invert), p.balance, p.contrast, p.brightness);
        if (p.use_texture != 0) {
            let t = tex_texture(c, p.texture_scale, p.texture_triplanar, p.texture_triplanar_contrast, 0,
                                p.texture_grayscale_conversion, p.texture_invert, p.texture_contrast, p.texture_brightness);
            m = g_blend(m, t.x, p.texture_blending_mode, p.texture_opacity * t.y);
        }
        return vec4f(vec3f(m), 1.0);
    }
    return global_final(c);
}
