@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let dir = ll_dir(uv);
    let ctr = ll_dir(vec2f(0.5) + p.offset);
    let f = frame(ctr);
    let z = dot(dir, ctr);
    let bg = backdrop(p.enable_background_input != 0, load_background_image_input(c).rgb, p.background_gamma == 0,
                      p.background_color.rgb);
    var col = bg;
    if (z > 0.05) {
        // Tangent-plane coordinates (0.25 tan units = the unit shape at identity), through the matrix.
        let q = inv2(p.matrix) * (vec2f(dot(dir, f[0]), -dot(dir, f[1])) / z / 0.25);
        let soft = mix(0.5, 0.005, sat(p.shape_hardness));
        var e = max(abs(q.x), abs(q.y));
        if (p.shape == 1) { e = length(q); }
        let mask = 1.0 - smoothstep(1.0 - soft, 1.0, e);
        let local = q * 0.5 + 0.5;
        let img = sample_shape_image_input(clamp(local, vec2f(0.0), vec2f(1.0))).rgb;
        let rgb = light_rgb(p.shape_color_mode, p.color.rgb, p.shape_temperature, img, p.shape_image_input_gamma == 0, background_image(c));
        let hs = max(p.hotspot_size, 1e-3);
        let hd = length(local - p.hotspot_position) / hs;
        let hot = 1.0 - smoothstep(1.0 - max(p.hotspot_falloff, 1e-3), 1.0, hd);
        let emit = rgb * (exp2(p.shape_exposure) + hot * (exp2(p.hotspot_exposure) - 1.0));
        col = mix(bg, emit, mask);
    }
    store(c, vec4f(col, 1.0));
}
