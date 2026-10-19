@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    var rgb = v.rgb;
    if (p.color_space == 1) { rgb = srgb2lin(rgb); }
    // Warmer = the light of a lower temperature: 6500 K / 4 at +1, x4 at -1.
    var m = blackbody(6500.0 * exp2(-2.0 * p.temperature)) / blackbody(6500.0);
    m.g *= exp2(0.5 * p.magenta_green);
    m /= max(luma(m), 1e-4);
    rgb *= m;
    if (p.color_space == 1) { rgb = lin2srgb(rgb); }
    store(c, vec4f(rgb, v.a));
}
