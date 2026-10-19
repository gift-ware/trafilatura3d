// The histogram image: block average (Bilinear) or the block's centre texel (Nearest), in the
// distance space; w = weight (alpha, or 1 with Ignore alpha).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let cell = 1.0 / vec2f(d);
    var v = vec4f(0.0);
    if (p.downscale_filtering == 1) {
        v = sample_input((vec2f(c) + 0.5) * cell);
        v = texel_input(vec2i(floor((vec2f(c) + 0.5) * cell * vec2f(size_input()))));
    } else {
        for (var j = 0; j < 4; j++) {
            for (var i = 0; i < 4; i++) {
                v += sample_input((vec2f(c) + (vec2f(f32(i), f32(j)) + 0.5) / 4.0) * cell) / 16.0;
            }
        }
    }
    let w = select(v.a, 1.0, p.ignore_alpha != 0);
    store_small(c, vec4f(to_space(v.rgb), clamp(w, 0.0, 1.0)));
}
