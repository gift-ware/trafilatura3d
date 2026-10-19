@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dir = ll_dir(uv_of(c, d));
    let zup = p.up_vector == 0;
    let cam = to_yup(p.camera_position, zup);  // the panorama's viewpoint
    let p1 = -cam + place(p.position_mode, p.point_1_uv_position, p.line_absolute_height, p.distance_from_origin,
                   p.point_1_world_position, zup);
    let p2 = -cam + place(p.position_mode, p.point_2_uv_position, p.line_absolute_height, p.distance_from_origin,
                   p.point_2_world_position, zup);
    let mid = (p1 + p2) * 0.5;
    let axis = p2 - p1;
    let len = max(length(axis), 1e-4);
    let a = axis / len;
    // Card normal: towards the origin, orthogonal to the line, turned around it.
    var n = -mid - a * dot(-mid, a);
    if (length(n) < 1e-5) { n = frame(a)[0]; }
    n = rot_axis(a, p.line_rotation * TAU) * normalize(n);
    let across = cross(n, a);
    let half_w = max(p.line_thickness * 0.25, 1e-4);
    var col = backdrop(p.enable_background_input != 0, load_background_image_input(c).rgb, p.background_gamma == 0,
                       p.background_color.rgb);
    let den = dot(dir, n);
    if (abs(den) > 1e-5) {
        let t = dot(mid, n) / den;
        let hit = dir * t;
        let s = dot(hit - p1, a) / len;
        let w = dot(hit - mid, across) / half_w;
        let clipped = p.enable_ground_clipping != 0 && hit.y < p.ground_height;
        if (t > 0.0 && s >= 0.0 && s <= 1.0 && abs(w) <= 1.0 && !clipped) {
            let aspect = len / (2.0 * half_w);
            let pu = along_uv(p.pattern_uv_mode, s, aspect, p.pattern_repeat_spacing);
            let pat_img = sample_pattern_image_input(vec2f(max(pu, 0.0), w * 0.5 + 0.5)).r;
            let mask = pattern_mask(p.pattern, vec2f(pu * 2.0 - 1.0, w), p.pattern_hardness, pat_img) * step(0.0, pu);
            let iu = along_uv(p.shape_image_uv_mode, s, aspect, p.shape_image_repeat_spacing);
            let img = sample_shape_image_input(vec2f(max(iu, 0.0), w * 0.5 + 0.5)).rgb * step(0.0, iu);
            let rgb = light_rgb(p.shape_color_mode, p.color.rgb, p.temperature, img, p.shape_image_gamma == 0, background_image(c));
            col = mix(col, rgb * exp2(p.exposure), mask);
        }
    }
    if (p.show_ground_grid != 0) { col += vec3f(0.6) * ground_grid(dir, p.ground_height); }
    store(c, vec4f(col, 1.0));
}
