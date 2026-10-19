// ---- scan/_atlas.wgsl: reading the detected shapes (inputs lbl, table, list) ----
fn shape_count() -> i32 { return i32(textureLoad(src_list, vec2i(0, 0), 0).y); }
fn shape_entry(j: i32) -> i32 { return i32(textureLoad(src_list, vec2i(clamp(j, 0, 255), 0), 0).x); }
fn shape_bbox(k: i32) -> vec4f { return textureLoad(src_table, vec2i(max(k, 0), 0), 0); }
fn shape_root(k: i32) -> vec2f { return textureLoad(src_table, vec2i(max(k, 0), 1), 0).zw; }
// Whether the atlas point belongs to the component of `root`.
fn in_shape(auv: vec2f, root: vec2f) -> bool {
    if (any(auv < vec2f(0.0)) || any(auv > vec2f(1.0))) { return false; }
    let r = size_lbl();
    let q = clamp(vec2i(auv * vec2f(r)), vec2i(0), r - 1);
    let l = textureLoad(src_lbl, q, 0).xy;
    return all(l == root);
}
