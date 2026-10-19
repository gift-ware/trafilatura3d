@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    // An HSL node set to the target's HSL minus the source's.
    store(c, vec4f(hsl_shift(v.rgb, rgb_to_hsl(p.target_color.rgb) - rgb_to_hsl(p.source_color.rgb)), v.a));
}
