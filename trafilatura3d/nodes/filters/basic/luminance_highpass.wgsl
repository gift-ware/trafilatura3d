@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    // The low frequencies of the luminance replaced by its mean: the same amount added to every channel.
    let shift = luma709(texel_mean(vec2i(0)).rgb) - luma709(load_blurred(c).rgb);
    store(c, vec4f(clamp(v.rgb + shift, vec3f(0.0), vec3f(1.0)), v.a));
}
