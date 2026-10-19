// Smoothed selection: a pixel stays selected where most of its gaussian neighbourhood is (the 0.5 contour of
// the blurred selection), antialiased over about half a pixel (contour distance = (v - 0.5) / |gradient|; the
// width is fitted on the output).
fn majority_cut(c: vec2i) -> f32 {
    let g = vec2f(texel_selection(c + vec2i(1, 0)).r - texel_selection(c - vec2i(1, 0)).r,
                  texel_selection(c + vec2i(0, 1)).r - texel_selection(c - vec2i(0, 1)).r) * 0.5;
    return clamp((texel_selection(c).r - 0.5) / max(0.6 * length(g), 1e-6) + 0.5, 0.0, 1.0);
}
