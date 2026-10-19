@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dir = ll_dir(uv_of(c, d));
    let zup = p.up_vector == 0;
    let ctr = -to_yup(p.camera_position, zup) + place(p.position_mode, p.plane_uv_position, p.plane_absolute_height, p.distance_from_origin,
                    p.plane_world_position, zup);
    // Card frame: flat for ground/ceiling, else facing the origin; spun by Plane Rotation.
    var n = -normalize(ctr + vec3f(0.0, 1e-5, 0.0));
    if (p.position_mode == 0) { n = vec3f(0.0, -sign(ctr.y + 1e-6), 0.0); }
    let f = frame(n);
    // Plane Rotation: x spins the card around its normal, y and z tilt it around its own axes.
    let tx0 = rot_axis(n, p.plane_rotation.x * TAU) * f[0];
    let tilt = rot_axis(cross(n, tx0), p.plane_rotation.z * TAU) * rot_axis(tx0, p.plane_rotation.y * TAU);
    n = tilt * n;
    let tx = tilt * tx0;
    let ty = cross(n, tx);
    let half = max(p.plane_scale * 2.0 * max(p.plane_size, vec2f(1e-3)), vec2f(1e-4));
    var col = backdrop(p.enable_background_input != 0, load_background_image_input(c).rgb, p.background_gamma == 0,
                       p.background_color.rgb);
    let den = dot(dir, n);
    if (abs(den) > 1e-5) {
        let t = dot(ctr, n) / den;
        let hit = dir * t;
        let q = vec2f(dot(hit - ctr, tx), dot(hit - ctr, ty)) / half;
        let clipped = p.enable_ground_clipping != 0 && hit.y < p.ground_height;
        if (t > 0.0 && max(abs(q.x), abs(q.y)) <= 1.0 && !clipped) {
            let aspect = half.x / half.y;
            let s = q.x * 0.5 + 0.5;
            let pu = along_uv(p.pattern_uv_mode, s, aspect, 0.0);
            let pat_img = sample_pattern_image_input(vec2f(pu, q.y * 0.5 + 0.5)).r;
            let mask = pattern_mask(p.pattern, vec2f(pu * 2.0 - 1.0, q.y), p.pattern_hardness, pat_img);
            let iu = along_uv(p.shape_image_uv_mode, s, aspect, p.shape_image_repeat_spacing);
            let img = sample_shape_image_input(vec2f(max(iu, 0.0), q.y * 0.5 + 0.5)).rgb * step(0.0, iu);
            let rgb = light_rgb(p.shape_color_mode, p.color.rgb, p.temperature, img, p.shape_image_gamma == 0, background_image(c));
            col = mix(col, rgb * exp2(p.exposure), mask);
        }
    }
    if (p.show_ground_grid != 0) { col += vec3f(0.6) * ground_grid(dir, p.ground_height); }
    store(c, vec4f(col, 1.0));
}
