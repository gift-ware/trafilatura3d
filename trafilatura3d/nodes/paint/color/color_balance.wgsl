fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let w = tone_weights(luminance(v.rgb));  // shadows, midtones, highlights
    let shift = w.x * vec3f(p.shadows_cyan_red, p.shadows_magenta_green, p.shadows_yellow_blue)
              + w.y * vec3f(p.midtones_cyan_red, p.midtones_magenta_green, p.midtones_yellow_blue)
              + w.z * vec3f(p.highlights_cyan_red, p.highlights_magenta_green, p.highlights_yellow_blue);
    return vec4f(clamp(v.rgb + shift * 0.4, vec3f(0.0), vec3f(1.0)), v.a);
}
