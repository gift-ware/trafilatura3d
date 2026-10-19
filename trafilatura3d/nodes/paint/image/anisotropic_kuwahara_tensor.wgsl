// Structure tensor (E, F, G) = (gx^2, gx gy, gy^2) of the direction source: Sobel gradients of the input's or the
// Custom Input's luminance, or the unit gradient across the Anisotropy Angle Map's direction (turns).
fn lum(c: vec2i) -> f32 {
    if (p.extract_direction == 1) { let t = load_custom_input(c); return select(luminance(t.rgb), t.r, CUSTOM_INPUT_GRAY); }
    let t = load_input(c);
    return select(luminance(t.rgb), t.r, INPUT_GRAY);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    var g = vec2f(0.0);
    if (p.extract_direction == 2) {
        let t = dir_of(load_anisotropy_angle_map(c).r);
        g = vec2f(-t.y, t.x);
    } else {
        let a = lum(c + vec2i(-1, -1)); let b = lum(c + vec2i(0, -1)); let d = lum(c + vec2i(1, -1));
        let e = lum(c + vec2i(-1, 0)); let f = lum(c + vec2i(1, 0));
        let h = lum(c + vec2i(-1, 1)); let i = lum(c + vec2i(0, 1)); let j = lum(c + vec2i(1, 1));
        g = vec2f((d + 2.0 * f + j) - (a + 2.0 * e + h), (h + 2.0 * i + j) - (a + 2.0 * b + d)) * 0.25;
    }
    store_out(c, vec4f(g.x * g.x, g.x * g.y, g.y * g.y, 1.0));
}
