@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dir = ll_dir(uv_of(c, d));
    var ctr = ll_dir(p.position) * max(p.distance_from_origin, 0.01);
    if (p.position_mode == 1) { ctr = to_yup(p.sphere_world_position, p.up_vector == 0); }
    ctr -= to_yup(p.camera_position, p.up_vector == 0);  // the panorama's viewpoint
    let bg = backdrop(p.enable_background_input != 0, load_background_image_input(c).rgb, p.background_gamma == 0,
                      p.background_color.rgb);
    let r = max(p.sphere_radius, 1e-4);
    let dist = length(ctr);
    var col = bg;
    // Angular size: soft edge over the outer (1 - hardness) of the disc as seen from the origin.
    let ang_r = select(PI, asin(min(r / max(dist, 1e-4), 1.0)), dist > r);
    let ang = acos(clamp(dot(dir, ctr / max(dist, 1e-4)), -1.0, 1.0));
    let edge = 1.0 - smoothstep(ang_r * mix(0.6, 0.995, sat(p.sphere_hardness)), ang_r, ang);
    if (edge > 0.0) {
        // Nearest hit of the ray (or the silhouette point when the soft edge reaches past it).
        let b = dot(dir, ctr);
        let disc = b * b - (dist * dist - r * r);
        let t = b - sqrt(max(disc, 0.0));
        let n = normalize(dir * t - ctr);
        var shade = 1.0;
        if (p.shading == 1) { shade = 1.0 - 0.6 * (1.0 - sat(dot(n, -dir))); }
        if (p.shading == 2) {
            let l = normalize(to_yup(p.shading_light_world_position, p.up_vector == 0) + vec3f(1e-5));
            let k = max(p.penombra_transparency, 0.01);
            shade = smoothstep(-k, k, dot(n, l));
        }
        // Globe mapping of the image: Sphere Rotation x spins it around its vertical axis, y tilts it.
        let m = rot_axis(vec3f(1.0, 0.0, 0.0), p.sphere_rotation.y * TAU) * rot_axis(vec3f(0.0, 1.0, 0.0), p.sphere_rotation.x * TAU) * n;
        let img = sample_shape_image_input(dir_ll(m)).rgb;
        let rgb = light_rgb(p.shape_color_mode, p.color.rgb, p.shape_temperature, img, p.sphere_image_input_gamma == 0, background_image(c));
        col = mix(bg, rgb * exp2(p.exposure) * shade, edge);
    }
    store(c, vec4f(col, 1.0));
}
