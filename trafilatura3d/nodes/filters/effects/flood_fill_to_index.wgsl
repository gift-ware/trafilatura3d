// Final pass: each pixel's shape index = kept first pixels before its shape's first pixel.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = size_special_shape_info();
    let q = ff_pixel(c, n);
    let s = texel_special_shape_info(q);
    let b = texel_flood_fill_bbox(ff_pixel(c, size_flood_fill_bbox()));
    let count = textureLoad(src_column, vec2i(0, size_column().y - 1), 0).r;
    store_shape_count(c, fx_gray(count));
    let kept = s.a > 0.0 && b.z > 0.0 && ff_big_enough(b.zw, p.ignore_shape_smaller_than);
    if (!kept) { store_shape_index(c, fx_gray(0.0)); return; }
    let l = vec2i(floor(s.xy * vec2f(n)));
    var before = 0.0;
    if (l.x > 0) { before += textureLoad(src_rows, vec2i(l.x - 1, l.y), 0).r; }
    if (l.y > 0) { before += textureLoad(src_column, vec2i(0, l.y - 1), 0).r; }
    store_shape_index(c, fx_gray(select(before / max(count, 1.0), before, p.output == 1)));
}
