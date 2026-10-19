// Entry j = index of the j-th shape whose bounding box diagonal reaches min_size; y = their count.
fn valid(k: i32) -> bool {
    let a = textureLoad(src_table, vec2i(k, 0), 0);
    let b = textureLoad(src_table, vec2i(k, 1), 0);
    return b.y > 0.0 && length(a.zw - a.xy) / 1.41421 >= p.min_size;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = i32(textureDimensions(src_table).x);
    var count = 0;
    var found = -1;
    for (var k = 0; k < n; k++) {
        if (valid(k)) {
            if (count == c.x) { found = k; }
            count++;
        }
    }
    store(c, vec4f(f32(found), f32(count), 0.0, 1.0));
}
