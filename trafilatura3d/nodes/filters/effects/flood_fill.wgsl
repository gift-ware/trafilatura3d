// Final pass: bounding box encoding, and the shape index from the prefix counts of first pixels.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let i = ff_index(c);
    if (atomicLoad(&parent[i]) == FF_NONE) {
        store_output(c, vec4f(0.0));
        store_special_shape_info(c, vec4f(0.0));
        return;
    }
    // The extents relative to the pixel, in the shape's frame.
    let s = ff_shape(i);
    let b = s.y;
    let f = (c - vec2i(ff_seam(atomicLoad(&box[b]), d.x, tile_h()), ff_seam(atomicLoad(&box[b + 1u]), d.y, tile_v()))
             + d) % d;
    let e = vec4f(f32(atomicLoad(&box[b + 2u])) - f32(f.x), f32(atomicLoad(&box[b + 3u])) - f32(f.x),
                  f32(atomicLoad(&box[b + 4u])) - f32(f.y), f32(atomicLoad(&box[b + 5u])) - f32(f.y));
    let ds = vec2f(d);
    let span = vec2f(e.y - e.x, e.w - e.z) + 1.0;
    let wh = min(span, ds);
    var local = (vec2f(-e.x, -e.z) + 0.5) / wh;
    // A shape wrapping all the way round an axis has no box on it: it spans the image.
    let wrapped = span >= ds;
    local = select(local, (vec2f(c) + 0.5) / ds, wrapped);
    store_output(c, vec4f(local, wh / ds));
    let l = ff_pixel(s.x);
    var before = 0.0;
    if (l.x > 0) { before += textureLoad(src_rows, vec2i(l.x - 1, l.y), 0).r; }
    if (l.y > 0) { before += textureLoad(src_column, vec2i(0, l.y - 1), 0).r; }
    let count = textureLoad(src_column, vec2i(0, d.y - 1), 0).r;
    store_special_shape_info(c, vec4f((vec2f(l) + 0.5) / ds, before, count));
}
