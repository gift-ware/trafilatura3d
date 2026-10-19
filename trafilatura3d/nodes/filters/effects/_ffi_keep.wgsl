// 1 on the first pixel of every shape kept (big enough), at the Special Shape Info's size.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = load_special_shape_info(c);
    let b = load_flood_fill_bbox(c);
    let root = vec2i(floor(s.xy * vec2f(d)));
    let kept = s.a > 0.0 && all(root == c) && ff_big_enough(b.zw, p.ignore_shape_smaller_than);
    store_out(c, fx_gray(select(0.0, 1.0, kept)));
}
