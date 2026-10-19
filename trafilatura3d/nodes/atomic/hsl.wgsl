@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    // Each slider adds 2 (value - 0.5) to its component.
    store(c, vec4f(hsl_shift(v.rgb, 2.0 * (vec3f(p.hue, p.saturation, p.lightness) - 0.5)), v.a));
}
