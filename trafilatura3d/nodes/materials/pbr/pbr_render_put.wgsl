@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    textureStore(dst_atlas, c + vec2i(0, p.row * d.y), vec4f(sample_src(uv_of(c, d)).rgb, 1.0));
}
