@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = wsn(load_world_space_normal(c).rgb);
    let h = p.horizontal_angle * TAU;
    let v = p.vertical_angle * TAU;
    let l = vec3f(-cos(v) * cos(h), sin(v), cos(v) * sin(h));
    // Half-Lambert window: Glossiness raises its start, Highlight Level pulls its end down onto the start.
    let x = 0.5 + 0.5 * dot(n, l);
    let g = p.highlight_glossiness;
    let top = mix(1.0, g, p.highlight_level);
    store(c, g4(sat((x - g) / max(top - g, 1e-4))));
}
