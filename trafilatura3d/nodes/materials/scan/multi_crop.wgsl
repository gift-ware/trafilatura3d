@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let q = crop_uv(uv);
    let inside = crop_inside(q);
    var bg = p.background;
    if (OUT_OUTPUT_1_GRAY) { bg = g4(bg.r); }
    var v: vec4f;
//@each
    v = sample_$ch(q);
    if (!$CH_GRAY && p.is_normal != 0) { v = vec4f(crop_normal(v.rgb, false), v.a); }
    if ($i >= p.input_count) { v = bg; }
    store_output_$n(c, select(bg, v, inside));
//@end
    store_area(c, crop_overlay(sample_input_1(uv), uv, p.show_crop_area != 0));
}
