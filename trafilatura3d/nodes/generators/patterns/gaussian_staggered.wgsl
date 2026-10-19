@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let g = frame(c, p.non_square_expansion) * f32(max(p.tiling, 1));
    // Two staggered rows of two blobs per tile, half the size of Gaussian 1's (sigma 19 px of 256).
    let v = min(gauss_rows(g, vec2f(0.5), vec2f(0.0, 0.25), 19.0 / 256.0), 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
