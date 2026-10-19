@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = size_flood_fill();
    let q = ff_pixel(c, n);
    let cell = ff_cell(texel_flood_fill(q), q, n);
    if (!cell.ok) { store(c, vec4f(0.0, 0.0, 0.0, 1.0)); return; }
    store(c, vec4f(ff_random(cell, 1.0), ff_random(cell, 2.0), ff_random(cell, 3.0), 1.0));
}
