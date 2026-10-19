fn effect(c: vec2i) -> vec4f {
    if (p.stage_ == 0) {  // R ambient occlusion (+ micro), G facing gradients, B thickness: before their blurs
        var a = load_baked_ambient_occlusion(c).r;
        a *= micro_occlusion(c, p.micro_normal, p.micro_height, p.curvature_type, p.curvature_intensity,
                             p.height_details_intensity, p.ao_radius, p.ao_depth);
        let n = normalize(g_normal(load_world_space_normals(c)) + vec3f(0.0, 1e-6, 0.0)) * 0.5 + 0.5;
        let w = g_three(n.zyx, vec3f(p.wsn_right_to_left, p.wsn_top_to_bottom, p.wsn_front_to_back),
                        vec3i(p.wsn_rl_invert, p.wsn_tb_invert, p.wsn_fb_invert),
                        vec3i(p.wsn_rl_blending_mode, p.wsn_tb_blending_mode, p.wsn_fb_blending_mode));
        return vec4f(g_invert(a, p.ao_invert), g_invert(w, p.wsn_invert),
                     g_invert(load_thickness(c).r, p.thickness_invert), 1.0);
    }
    if (p.stage_ == 1) {
        let uv = uv_of(c);
        var m = 0.0;
        let t1 = tex_texture(c, p.texture_scale, p.texture_triplanar, p.texture_triplanar_contrast,
                             p.texture_non_square_tiling, p.texture_grayscale_conversion, p.texture_invert,
                             p.texture_contrast, p.texture_brightness);
        m = g_blend(m, t1.x, p.texture_blending_mode, p.texture_opacity * t1.y);
        let t2 = tex_texture_2(c, p.texture_2_scale, p.texture_2_triplanar, p.texture_2_triplanar_contrast,
                               p.texture_2_non_square_tiling, p.texture_2_grayscale_conversion, p.texture_2_invert,
                               p.texture_2_contrast, p.texture_2_brightness);
        m = g_blend(m, t2.x, p.texture_2_blending_mode, p.texture_2_opacity * t2.y);
        let ao = g_adjust(sample_blur_r_(uv).r, p.ao_balance, p.ao_contrast, 0.0);
        m = g_blend(m, ao, p.ao_blending_mode, p.ambient_occlusion_opacity);
        let mc = micro_curvature(c, p.micro_normal, p.micro_height, p.curvature_type, p.curvature_intensity,
                                 p.height_details_intensity);
        var cv = g_curvature_mode(clamp(load_curv_(c).r + mc - 0.5, 0.0, 1.0), p.curvature_mode);
        cv = g_adjust(g_invert(cv, p.curvature_invert), 0.5, p.curvature_contrast, p.curvature_brightness);
        m = g_blend(m, cv, p.curvature_blending_mode, p.curvature_opacity);
        let wsn = g_adjust(sample_blur_g_(uv).g, p.wsn_balance, p.wsn_contrast, p.wsn_brightness);
        m = g_blend(m, wsn, p.wsn_blending_mode, p.world_space_normal_opacity);
        let pos = clamp(load_position(c).rgb, vec3f(0.0), vec3f(1.0));
        var g = g_three(pos.zyx, vec3f(p.position_right_to_left, p.position_top_to_bottom, p.position_front_to_back),
                        vec3i(p.position_rl_invert, p.position_tb_invert, p.position_fb_invert),
                        vec3i(p.position_rl_blending_mode, p.position_tb_blending_mode, p.position_fb_blending_mode));
        g = g_adjust(g_invert(g, p.position_invert), p.position_balance, p.position_contrast, p.position_brightness);
        m = g_blend(m, g, p.position_blending_mode, p.position_gradient_opacity);
        let th = g_adjust(sample_blur_b_(uv).b, 0.5, p.thickness_contrast, p.thickness_brightness);
        m = g_blend(m, th, 0, p.thickness_opacity);
        return vec4f(vec3f(m), 1.0);
    }
    return global_final(c);
}
