// Shared by the Shape Splatter passes.

const SS_LIMIT = 2.5;  // tiles stay within this many cells of their cell (the gather reach)

fn ss_counts() -> vec2i { return max(vec2i(p.x_amount, p.y_amount), vec2i(1)); }

// Half extent of a tile at scale 1, in uv: the cell, or towards a square of the cell's shorter side with
// Preserve Aspect Ratio, times Size.
fn ss_half0() -> vec2f {
    let cs = 1.0 / vec2f(ss_counts());
    return mix(cs, vec2f(min(cs.x, cs.y)), p.preserve_aspect_ratio) * 0.5 * p.size;
}

// Slope vector from a height gradient (per uv), length clamped to 1.
fn ss_slope(grad: vec2f) -> vec2f {
    let s = grad * 0.1;
    let l = length(s);
    return select(s, s / l, l > 1.0);
}

fn ss_global_offset() -> vec2f {
    return vec2f(p.offset_horizontal, p.offset_vertical) / vec2f(ss_counts());
}

// Upper bound of a tile's half extent plus displacement, in cells.
fn ss_extent() -> f32 {
    if (p.scale_map_multiplier > 0.0) { return SS_LIMIT; }
    let n = vec2f(ss_counts());
    let h = ss_half0() * n;
    let sc = p.scale * (1.0 + p.scale_random);
    return min(max(h.x, h.y) * sc * 1.415 + p.position_random * 0.5 + p.vector_map_multiplier, SS_LIMIT);
}
