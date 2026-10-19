// Quantize, palette pass 1: the input averaged down to a small grid (4x4 bilinear samples per cell), in the distance
// space. The palette is extracted from this grid, so it does not depend on the output size.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    var acc = vec3f(0.0);
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let uv = (vec2f(c) + (vec2f(f32(i), f32(j)) + 0.5) / 4.0) / p.size_;
            acc += sample_input(uv).rgb;
        }
    }
    store(c, vec4f(to_space(acc / 16.0, p.space_), 1.0));
}
