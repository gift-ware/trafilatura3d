@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    // Each channel pulled towards its surroundings by Intensity, only where it is above them.
    store(c, vec4f(min(v.rgb, mix(v.rgb, load_blurred(c).rgb, p.intensity)), v.a));
}
