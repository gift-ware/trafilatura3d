// Glaze of the bled paint over the paper; pigment density from the borders (difference with the spread paint)
// and the paper's grain.
fn paper_at(uv: vec2f, c: vec2i) -> f32 {
    if (p.custom_paper != 0) { return load_paper(c).r; }
    let fibers = mx_fbm(uv * vec2f(1.0, 1.0), 64, 4, 0.6, p.seed_ + 20.0);
    let grain = mx_noise(uv, vec2i(220), p.seed_ + 21.0);
    return mx_sat(0.55 + fibers * 0.3 + grain * 0.15);
}

fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let s = load_bleed(c);
    let b = sample_spread(uv);
    var edge = abs(s.a - b.a) * 2.0;
    if (p.show_cross_paint_edges != 0) { edge = max(edge, length(s.rgb - b.rgb) * 2.5); }
    edge = mx_sat(edge);
    var dens = 1.0;
    if (p.accumulation_type == 0) { dens = 1.0 + edge * p.borders_density * 3.0; }
    else if (p.accumulation_type == 1) { dens = 1.0 + (1.0 - edge) * p.borders_density * 1.5 - edge * 0.5; }
    var paper = 1.0;
    if (p.hide_paper == 0) {
        paper = paper_at(uv, c);
        dens *= mix(1.0, 1.5 - paper * 0.9, p.paper_absorption);  // pigment settles in the paper's valleys
    }
    let pig = pow(max(s.rgb, vec3f(0.002)), vec3f(max(dens, 0.1)));
    var col = mix(vec3f(1.0), pig, p.paint_opacity * s.a);
    // The paper tints everything and covers outside the paint; hidden, only the paint's coverage remains.
    var a = s.a;
    if (p.hide_paper == 0) { col *= mix(1.0, paper, 0.35); a = 1.0; }
    store_basecolor(c, vec4f(col, a));
}
