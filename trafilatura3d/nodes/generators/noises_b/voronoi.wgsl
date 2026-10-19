@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    var cells = max(p.scale * p.size, vec3f(1e-3));
    var per = vec3i(1 << 20);
    // Tiled axes: whole cells per tile (the stepped scale).
    if (tile_h()) { cells.x = max(round(cells.x), 1.0); per.x = i32(cells.x); }
    if (tile_v()) { cells.y = max(round(cells.y), 1.0); per.y = i32(cells.y); }
    let r = voronoi3(vpoint(uv, cells), per, p.seed_, color_seed());
    var v = sat(vstyle(r));
    if (p.invert != 0) { v = 1.0 - v; }
    store(c, vec4f(v, v, v, 1.0));
}
