@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let n = 2.0 * f32(max(p.tiling, 1));
    let v = mesh_cells(grid(uv, n, p.rotate_45_degrees), false);
    store(c, vec4f(v, v, v, 1.0));
}
