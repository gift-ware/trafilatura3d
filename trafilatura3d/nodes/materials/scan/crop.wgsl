fn take(q: vec2f) -> vec4f {
    if (p.filtering == 1) { return sample_input((floor(q * vec2f(size_input())) + 0.5) / vec2f(size_input())); }
    return sample_input(q);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let q = crop_uv(uv);
    var v = take(q);
    if (!INPUT_GRAY && p.is_normal != 0) { v = vec4f(crop_normal(v.rgb, p.normal_format == 0), v.a); }
    var bg = p.background;
    if (OUT_OUTPUT_GRAY) { bg = g4(bg.r); }
    store_output(c, select(bg, v, crop_inside(q)));
    store_area(c, crop_overlay(sample_input(uv), uv, p.show_crop_area != 0));
}
