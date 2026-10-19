// Flood fill: each white pixel points straight to its root.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let i = ff_index(c);
    if (atomicLoad(&parent[i]) == FF_NONE) { return; }
    atomicStore(&parent[i], ff_find(i));
}
