// ---- filters/_hblend.wgsl: height-based blend (Height Blend, Material Height Blend), ours ----
// Height Offset favours one height, s = 2 Offset - 1 in -1 .. 1 (0.5 = neither). Below 0.5 the top is scaled down
// by 1 + s. Above it, Balanced height scales the bottom down by 1 - s, Bottom height priority raises the top
// towards 1 by s. The user mask scales the top and the bottom's lowering. Returns (top, bottom).
fn hb_heights(top: f32, bottom: f32, offset: f32, mode: i32, mask: f32) -> vec2f {
    let s = 2.0 * offset - 1.0;
    var t = top * (1.0 + min(s, 0.0));
    var b = bottom;
    if (mode == 0) {
        b = bottom * (1.0 - max(s, 0.0) * mask);
    } else {
        t = mix(t, 1.0, max(s, 0.0));
    }
    return vec2f(t * mask, b);
}

// The mask of where the top shows: Opacity where the top is at or above the bottom, falling linearly to 0 where it
// is 1 - Contrast below it (Contrast 1: a hard edge).
fn hb_mask_of(h: vec2f, contrast: f32, opacity: f32) -> f32 {
    let w = 1.0 - sat(contrast);
    let diff = h.x - h.y;
    if (w < 1e-5) { return opacity * select(0.0, 1.0, diff >= 0.0); }
    return opacity * sat(1.0 + diff / w);
}

// The mask without a user mask (Material Height Blend).
fn hb_mask(top: f32, bottom: f32, offset: f32, contrast: f32, mode: i32, opacity: f32) -> f32 {
    return hb_mask_of(hb_heights(top, bottom, offset, mode, 1.0), contrast, opacity);
}
