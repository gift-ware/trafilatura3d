@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    var g = 0.15 + 0.1 * uv.x;                                              // gentle slope
    g = max(g, smoothstep(0.22, 0.12, length(uv - vec2f(0.3, 0.3))));      // soft disc
    g = max(g, 0.7 * smoothstep(0.16, 0.06, length(uv - vec2f(0.72, 0.62))));
    let sq = step(abs(uv.x - 0.3), 0.12) * step(abs(uv.y - 0.75), 0.12);  // hard square
    g = mix(g, 0.9, sq);
    let band = smoothstep(0.55, 0.6, uv.y) * smoothstep(0.95, 0.9, uv.y) * step(0.5, uv.x);
    g = mix(g, uv.x, band * 0.8);                                           // gradient band
    let cell = vec2i(floor(uv * 32.0));
    g = clamp(g + (hash21(cell, 3.0) - 0.5) * 0.08, 0.0, 1.0);             // fine detail
    if (p.color == 1) {
        let tint = vec3f(0.5 + 0.5 * cos(6.2831 * (uv.x + vec3f(0.0, 0.33, 0.67))));
        store(c, vec4f(g * mix(vec3f(1.0), tint, 0.7), 1.0));
    } else {
        store(c, vec4f(g, g, g, 1.0));
    }
}
