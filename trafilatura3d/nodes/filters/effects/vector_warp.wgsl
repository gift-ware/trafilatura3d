@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    var v = load_vector_map(c).rg * 2.0 - 1.0;
    if (p.vector_format == 1) { v.y = -v.y; }  // OpenGL: green up, rows go down
    // The input read at uv + v |v| Intensity (the vector scaled by its own length).
    store(c, sample_input(uv + v * length(v) * p.intensity));
}
