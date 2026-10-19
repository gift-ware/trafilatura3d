@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let e = 1.0 / vec2f(d);
    let gx = sample_intensity_input(uv + vec2f(e.x, 0.0)).r - sample_intensity_input(uv - vec2f(e.x, 0.0)).r;
    let gy = sample_intensity_input(uv + vec2f(0.0, e.y)).r - sample_intensity_input(uv - vec2f(0.0, e.y)).r;
    let a = p.light_angle * 2.0 * PI;
    // A slope rising towards the light faces it.
    let s = -dot(vec2f(gx, gy), vec2f(cos(a), sin(a))) * p.intensity * f32(d.x) / 256.0;
    let v = load_input(c);
    // The highlight adds its color, the shadow multiplies by its color: black / white turn the light off.
    let lit = v.rgb + p.highlight_color.rgb * clamp(s, 0.0, 1.0);
    var o = vec4f(lit * mix(vec3f(1.0), p.shadow_color.rgb, clamp(-s, 0.0, 1.0)), v.a);
    if (INPUT_GRAY) { let g = luminance(o.rgb); o = vec4f(g, g, g, 1.0); }
    store(c, vec4f(o.rgb, v.a));
}
