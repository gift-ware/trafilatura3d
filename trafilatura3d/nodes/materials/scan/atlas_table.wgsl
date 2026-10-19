// One thread per table entry k: the k-th root in raster order, then its component's bounding box.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let k = c.x;
    let r = size_lbl();
    var count = -1;
    var root = vec2i(-1);
    for (var y = 0; y < r.y && root.x < 0; y++) {
        for (var x = 0; x < r.x; x++) {
            let l = textureLoad(src_lbl, vec2i(x, y), 0);
            if (i32(l.x) == x && i32(l.y) == y) {
                count++;
                if (count == k) { root = vec2i(x, y); break; }
            }
        }
    }
    if (root.x < 0) {
        textureStore(dst_table, vec2i(k, 0), vec4f(0.0));
        textureStore(dst_table, vec2i(k, 1), vec4f(0.0, 0.0, -1.0, -1.0));
        return;
    }
    var lo = vec2i(r);
    var hi = vec2i(-1);
    var area = 0.0;
    for (var y = root.y; y < r.y; y++) {
        for (var x = 0; x < r.x; x++) {
            let l = textureLoad(src_lbl, vec2i(x, y), 0);
            if (i32(l.x) == root.x && i32(l.y) == root.y) {
                lo = min(lo, vec2i(x, y));
                hi = max(hi, vec2i(x, y));
                area += 1.0;
            }
        }
    }
    let rs = vec2f(r);
    textureStore(dst_table, vec2i(k, 0), vec4f(vec2f(lo) / rs, vec2f(hi + 1) / rs));
    textureStore(dst_table, vec2i(k, 1), vec4f(area / (rs.x * rs.y), 1.0, f32(root.x), f32(root.y)));
}
