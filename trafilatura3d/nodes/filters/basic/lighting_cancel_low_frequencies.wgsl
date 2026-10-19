@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    // The slow changes (the local mean) replaced by the image's mean colour; the detail applied as an exposure
    // (light is multiplicative): darker detail compressed, brighter detail opened up.
    // Only the detail's luminance is an exposure, its colour is added as it is (no tint pushed by the mean).
    let detail = v.rgb - load_blurred(c).rgb;
    let l = luma709(detail);
    let o = texel_mean(vec2i(0)).rgb * exp(p.gain * l) + (detail - l);
    store(c, vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), v.a));
}
