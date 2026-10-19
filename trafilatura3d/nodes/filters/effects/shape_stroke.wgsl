@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let shape = shape_layer(c, p.mask_color, p.input_is_pre_multiplied == 1);
    let outer = p.width >= 0.0;
    let reach = max(shape_reach(abs(p.width)), 1e-3);
    let x = select(-shape_sd(c), shape_sd(c), outer);  // distance away from the edge, on the stroke's side
    let band = clamp(reach - max(x, 0.0) + 0.5, 0.0, 1.0);  // solid out to the reach, a pixel of antialiasing
    let stroke = clamp(band - select(1.0 - shape.a, shape.a, outer), 0.0, 1.0);
    let k = p.opacity * p.outline_color.a;
    var o: vec4f;
    if (outer) { o = over(shape, vec4f(p.outline_color.rgb, band * k)); }
    else { o = over(vec4f(p.outline_color.rgb, stroke * k), shape); }
    store_output(c, finish(o, p.pre_multiply_output == 1));
    store_mask(c, fx_gray(stroke));
}
