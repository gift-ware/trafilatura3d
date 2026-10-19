@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = size_flood_fill();
    let q = ff_pixel(c, n);
    let cell = ff_cell(texel_flood_fill(q), q, n);
    var v = max(cell.size.x, cell.size.y);
    if (p.output == 1) { v = cell.size.x; }
    if (p.output == 2) { v = cell.size.y; }
    store(c, fx_gray(select(0.0, v, cell.ok)));
}
