@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let k = p.output_color;
    if (OUT_OUTPUT_GRAY) { store(c, vec4f(k.r, k.r, k.r, 1.0)); } else { store(c, k); }
}
