// The last column of the row prefix sums (one value per row), into a 1-pixel-wide texture.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    store_out(c, fx_gray(textureLoad(src_rows, vec2i(size_rows().x - 1, c.y), 0).r));
}
