@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_basecolor(c);
    let ao = clamp(load_ambient_occlusion(c).r, 0.05, 1.0);
    // Work in linear light: lighting multiplies the albedo there.
    let gain = 1.0 / mix(1.0, ao, p.ao_cancellation);
    var lin = srgb2lin(v.rgb) * gain;
    let lift = sat((gain - 1.0) * 0.5);
    lin = saturate_color(lin, 1.0 + p.ao_saturation * lift * 2.0);
    store(c, vec4f(lin2srgb(max(lin, vec3f(0.0))), v.a));
}
