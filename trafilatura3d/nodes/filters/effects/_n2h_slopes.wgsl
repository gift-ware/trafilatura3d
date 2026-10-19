// Slopes of the surface a normal map describes: dh/du, dh/dv (rows down) in height per UV.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = fx_decode_normal(load_input(c), p.directx == 1);
    let z = max(n.z, 0.1);
    store_out(c, vec4f(-n.x / z, n.y / z, 0.0, 1.0));
}
