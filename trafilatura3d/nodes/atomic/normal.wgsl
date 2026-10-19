@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let e = 1.0 / vec2f(d);
    var h: array<f32, 9>;
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) {
            h[j * 3 + i] = sample_input(uv + vec2f(f32(i - 1), f32(j - 1)) * e).r;
        }
    }
    let dx = (h[2] + 2.0 * h[5] + h[8]) - (h[0] + 2.0 * h[3] + h[6]);
    let dy = (h[0] + 2.0 * h[1] + h[2]) - (h[6] + 2.0 * h[7] + h[8]);  // rows grow downwards: +Y is up
    // Slopes per 256 px pixel, so the relief holds at every resolution.
    let k = p.intensity * f32(d.x) / 256.0 / 8.0;
    var n = normalize(vec3f(-dx * k, -dy * k, 1.0));
    if (p.normal_format == 0) { n.y = -n.y; }  // DirectX: green flipped
    let a = select(h[4], 1.0, p.alpha_channel_content == 1);
    store(c, vec4f(n * 0.5 + 0.5, a));
}
