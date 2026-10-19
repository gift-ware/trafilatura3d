// Unit chroma direction of an OKLab color (its hue), `fallback` for a gray.
fn hue_dir(lab: vec3f, fallback: vec2f) -> vec2f {
    let n = length(lab.yz);
    return select(fallback, lab.yz / n, n > 1e-4);
}

fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    var src = texel_mean(vec2i(0)).rgb;
    let src_in = load_source_color(c).rgb;
    if (p.source_color_mode == 1) { src = p.source_color; }
    if (p.source_color_mode == 2) { src = src_in; }
    let dst_in = load_target_color(c).rgb;
    let dst = select(p.target_color, dst_in, p.target_color_mode == 1);
    let s = srgb_to_oklab(src);
    let t = srgb_to_oklab(dst);
    let x = srgb_to_oklab(v.rgb);
    // The pixel's deviation from the source in the source's frame: lightness, chroma (along its hue), hue (across).
    let us = hue_dir(s, vec2f(1.0, 0.0));
    let vs = vec2f(-us.y, us.x);
    let ut = hue_dir(t, us);
    let vt = vec2f(-ut.y, ut.x);
    let d = x - s;
    var k = vec3f(1.0);  // luma, chroma, hue
    if (p.custom_color_variation != 0) { k = vec3f(p.luma, p.chroma, p.hue); }
    let ab = t.yz + ut * dot(d.yz, us) * k.y + vt * dot(d.yz, vs) * k.z;
    return vec4f(oklab_to_srgb(vec3f(t.x + d.x * k.x, ab)), v.a);
}
