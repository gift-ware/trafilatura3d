// Flood fill, first pass: every white pixel (input cut at 0.5) points to its left neighbour when that one is white
// (inside the image), else to itself; black is FF_NONE.
fn ff_input_white(q: vec2i) -> bool { return load_input(q).r >= 0.5; }  // resampled: the input may have another size

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let i = ff_index(c);
    if (!ff_input_white(c)) { atomicStore(&parent[i], FF_NONE); return; }
    atomicStore(&parent[i], select(i, i - 1u, c.x > 0 && ff_input_white(c - vec2i(1, 0))));
}
