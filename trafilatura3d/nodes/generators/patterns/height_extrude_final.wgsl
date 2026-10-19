@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var h: array<f32, 9>;
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) { h[j * 3 + i] = texel_depth_map(c + vec2i(i - 1, j - 1)).r; }
    }
    let dx = (h[2] + 2.0 * h[5] + h[8]) - (h[0] + 2.0 * h[3] + h[6]);
    let dy = (h[0] + 2.0 * h[1] + h[2]) - (h[6] + 2.0 * h[7] + h[8]);
    let k = p.normal_intensity * f32(d.x) / 256.0 / 8.0;
    var n = normalize(vec3f(-dx * k, -dy * k, 1.0));
    if (p.normal_format == 0) { n.y = -n.y; }
    store_depth(c, vec4f(h[4], h[4], h[4], 1.0));
    store_normal(c, vec4f(n * 0.5 + 0.5, 1.0));
}
