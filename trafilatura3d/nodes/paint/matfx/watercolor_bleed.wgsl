// The paint bleeding into the paper: the base color read through a noise warp (stronger with absorption).
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let w = vec2f(mx_fbm(uv, 10, 4, 0.55, p.seed_ + 1.0), mx_fbm(uv, 10, 4, 0.55, p.seed_ + 2.0)) - 0.5;
    store_bleed(c, sample_basecolor(uv + w * (0.004 + p.paper_absorption * 0.012)));
}
