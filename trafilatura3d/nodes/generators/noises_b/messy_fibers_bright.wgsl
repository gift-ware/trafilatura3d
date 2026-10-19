@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion) + p.tile_offset;
    let per = max(p.scale, 1);
    let v = fibers(uv, per, 2, 14, 1.2, 0.14, 2.0 * p.lines_number, 8.0, 0.35, false, p.seed_);
    store(c, vec4f(v, v, v, 1.0));
}
