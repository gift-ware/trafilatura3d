fn address(x: vec4f) -> vec4f {
    if (p.curve_addressing == 1) {  // fold: 1.2 -> 0.8, -0.2 -> 0.2
        let m = abs(x) % vec4f(2.0);
        return select(m, 2.0 - m, m > vec4f(1.0));
    }
    return clamp(x, vec4f(0.0), vec4f(1.0));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var v = load_input(c);
    if (p.expose_curve != 0) { v = vec4f(vec3f((f32(c.x) + 0.5) / f32(d.x)), 1.0); }
    var x = address(v);
    if (!INPUT_GRAY || p.expose_curve != 0) {
        x = vec4f(lut_curve_red(x.r).r, lut_curve_green(x.g).r, lut_curve_blue(x.b).r, lut_curve_alpha(x.a).r);
    }
    let y = vec4f(lut_curve(x.r).r, lut_curve(x.g).r, lut_curve(x.b).r, x.a);
    if (INPUT_GRAY && p.expose_curve == 0) { store(c, vec4f(y.r, y.r, y.r, 1.0)); } else { store(c, y); }
}
