@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let q = crop_uv(uv);
    let inside = crop_inside(q);
    var v: vec4f;
    var bg: vec4f;
//@each
    v = sample_$ch(q);
    bg = p.background;
    if ($K == 1) { v = vec4f(crop_normal(v.rgb, p.normal_format == 0), 1.0); bg = vec4f(0.5, 0.5, 1.0, 1.0); }
    if ($K == 2) { v = g4(v.r); bg = g4(bg.r); }
    store_$ch(c, out_$ch(c, select(bg, v, inside)));
//@end
    store_area(c, crop_overlay(sample_basecolor(uv), uv, p.show_crop_area != 0));
}
