// Hexagonal hue in [0, 6) from the largest channel; 0 for a gray.
fn hex_hue(c: vec3f, chroma: f32) -> f32 {
    if (chroma <= 0.0) { return 0.0; }
    let m = max(c.r, max(c.g, c.b));
    if (m == c.r) { return ((c.g - c.b) / chroma + 6.0) % 6.0; }
    if (m == c.g) { return (c.b - c.r) / chroma + 2.0; }
    return (c.r - c.g) / chroma + 4.0;
}

// The colour of hexagonal hue h (one sector per unit) at chroma `ch`, lowest channel 0. Below the circle nothing
// (gray), past it the last edge reaches red.
fn hex_color(h: f32, ch: f32) -> vec3f {
    if (h < 0.0) { return vec3f(0.0); }
    let x = ch * (1.0 - abs(h % 2.0 - 1.0));
    if (h < 1.0) { return vec3f(ch, x, 0.0); }
    if (h < 2.0) { return vec3f(x, ch, 0.0); }
    if (h < 3.0) { return vec3f(0.0, ch, x); }
    if (h < 4.0) { return vec3f(0.0, x, ch); }
    if (h < 5.0) { return vec3f(x, 0.0, ch); }
    if (h < 6.0) { return vec3f(ch, 0.0, x); }
    return vec3f(ch, 0.0, 0.0);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    var chroma = max(v.r, max(v.g, v.b)) - min(v.r, min(v.g, v.b));
    var y = luma709(v.rgb);
    var h = hex_hue(v.rgb, chroma) / 6.0 + select(p.hue_turns, p.hue_degrees / 360.0, p.hue_mode == 1);
    if (h > 1.0) { h -= 1.0; } else if (h < 0.0) { h += 1.0; }
    if (p.proportional != 0) {
        let k = p.proportional_chrominance / 100.0;
        let l = p.proportional_luminance / 100.0;
        chroma = select(chroma * (1.0 + k), chroma + k * (1.0 - chroma), k > 0.0);
        y = select(y * (1.0 + l), y + l * (1.0 - y), l > 0.0);
    } else {
        chroma += p.chrominance;
        y += p.luminance;
    }
    if (p.clamp_chroma != 0) { chroma = max(chroma, 0.0); }
    let o = hex_color(h * 6.0, chroma);
    store(c, vec4f(clamp(o + (y - luma709(o)), vec3f(0.0), vec3f(1.0)), v.a));
}
