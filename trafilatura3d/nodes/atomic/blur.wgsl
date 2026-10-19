@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dir = vec2f(f32(1 - p.vertical), f32(p.vertical));
    // Pixels at 256 times Pixel Size along the pass's axis.
    let radius = max(p.intensity * dot(dir, p.pixel_size_), 0.0) * f32(d.x) / 256.0;
    let r = i32(ceil(radius));
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    var acc = vec4f(0.0);
    var wsum = 0.0;
    for (var k = -r; k <= r; k++) {
        // Fractional radius: the outermost taps weigh what is left of it.
        let w = clamp(radius + 0.5 - abs(f32(k)), 0.0, 1.0);
        acc += w * sample_input(uv + dir * f32(k) / vec2f(d));
        wsum += w;
    }
    store(c, acc / max(wsum, 1e-5));
}
