// First finishing pass of every grunge map: sharpen (unsharp mask, radius one pixel at 256 px), the default
// Histogram Scan window undone (the raw pattern into [window / 2, 1 - window / 2]), Invert, and the Brush
// Pattern edge mask. Balance and Contrast follow (Histogram Scan).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    var v = sample_input(uv).r;
    if (p.sharpen > 0.0) {
        let e = 1.0 / 256.0;
        let blur = (sample_input(uv + vec2f(e, 0.0)).r + sample_input(uv - vec2f(e, 0.0)).r +
                    sample_input(uv + vec2f(0.0, e)).r + sample_input(uv - vec2f(0.0, e)).r) * 0.25;
        v = v + (v - blur) * p.sharpen;
    }
    v = 0.5 * p.window + sat(v) * (1.0 - p.window);
    if (p.invert != 0) { v = 1.0 - v; }
    if (p.brush > 0.0) {
        let r = length(uv - 0.5);
        let edge = (0.5 - r) / (0.5 * p.brush);
        v *= smoothstep(0.0, 1.0, sat(edge));
    }
    store(c, vec4f(v, v, v, 1.0));
}
