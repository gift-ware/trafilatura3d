@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let shape = shape_layer(c, p.mask_color, p.input_is_pre_multiplied == 1);
    let outer = p.width >= 0.0;
    var b = clamp(texel_blur(c).r, 0.0, 1.0);
    if (p.mode == 0) { b = pow(b, 0.44); }  // Soft: a brighter, longer tail (gamma fitted on the output)
    let g = clamp(b / max(1.0 - p.spread, 1e-3), 0.0, 1.0);
    let k = p.opacity * p.glow_color.a;
    var o: vec4f;
    if (outer) { o = over(shape, vec4f(p.glow_color.rgb, g * k)); }
    else { o = over(vec4f(p.glow_color.rgb, g * k * shape.a), shape); }
    store_output(c, finish(o, p.pre_multiply_output == 1));
    store_mask(c, fx_gray(g * select(shape.a, 1.0 - shape.a, outer)));
}
