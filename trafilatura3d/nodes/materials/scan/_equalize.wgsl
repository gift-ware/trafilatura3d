// ---- scan/_equalize.wgsl: Color Equalizer core ----
// 1 where `v` is within Hue Range / 2 degrees of `r`'s hue, Chroma Range / 2 of its chroma (max - min)
// and Luma Range / 2 of its luma (Rec. 601), else 0.
fn color_mask(v: vec3f, r: vec3f) -> f32 {
    let dh = abs(fract(rgb2hsv(sat3(v)).x - rgb2hsv(sat3(r)).x + 0.5) - 0.5) * 360.0;
    let dc = (max(v.r, max(v.g, v.b)) - min(v.r, min(v.g, v.b))) - (max(r.r, max(r.g, r.b)) - min(r.r, min(r.g, r.b)));
    let dl = dot(v - r, vec3f(0.299, 0.587, 0.114));
    return select(0.0, 1.0, dh <= p.hue_range * 0.5 && abs(dc) <= p.chroma_range * 0.5 && abs(dl) <= p.luma_range * 0.5);
}

// The mask of Mask Source (before its Blur): around the image average (None, Image Average), around
// Color (Color Parameter), or the Mask Input cut at 0.5 (Input).
fn equalizer_mask(v: vec3f, avg: vec3f, ext: f32) -> f32 {
    switch p.mask_source {
        case 2: { return color_mask(v, p.color.rgb); }
        case 3: { return step(0.5, ext); }
        default: { return color_mask(v, avg); }
    }
}

// v: the pixel, b: its blurred neighbourhood, avg: image average, m: the mask (not applied with
// Mask Source None); returns the equalized color.
fn equalize(v: vec3f, b: vec3f, avg: vec3f, m: f32) -> vec3f {
    var tgt = avg;
    if (p.custom_color_variation != 0) { tgt = hsv2rgb(vec3f(p.hue / 360.0, p.chroma, p.luma)); }
    // Balance: 0 keeps the darker tints, 1 removes them most.
    let lb = luma(b);
    let w = sat(mix(lb * 2.0, 2.0 - lb * 2.0, p.bright_dark_balance) * 0.5 + 0.5);
    return v + (tgt - b) * w * select(m, 1.0, p.mask_source == 0);
}
