@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let g = vec2f(vol_grid());
    let at = (vec2f(c) + 0.5) / vec2f(d) * g;  // uv x Grid Size
    let cell = floor(at);
    store(c, vec4f(fract(at), (cell.y * g.x + cell.x) / (g.x * g.y), 1.0));
}
