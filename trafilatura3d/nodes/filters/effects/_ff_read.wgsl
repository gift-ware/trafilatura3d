// Reading Flood Fill data (R, G = position in the bounding box, B, A = box size in UV; black = no
// shape) for the output pixel c, from an input of size `n` read with `t` at the matching pixel `q`.
struct FfCell {
    ok: bool,
    centre: vec2f,  // box centre, UV (wrapped into 0-1)
    size: vec2f,    // box size, UV
    local: vec2f,   // position inside the box, 0-1
    key: vec2i,     // the same for every pixel of a shape: its centre on a half-pixel grid
}

fn ff_pixel(c: vec2i, n: vec2i) -> vec2i { return vec2i(floor((vec2f(c) + 0.5) * vec2f(n) / p.size_)); }

fn ff_cell(t: vec4f, q: vec2i, n: vec2i) -> FfCell {
    var r: FfCell;
    r.ok = t.z > 0.0 && t.w > 0.0;
    let nf = vec2f(n);
    let wh = t.zw * nf;
    let centre_px = vec2f(q) + 0.5 + (0.5 - t.xy) * wh;
    r.key = vec2i(round(centre_px * 2.0));
    r.key = wrap(r.key, 2 * n);
    r.centre = fract(centre_px / nf);
    r.size = t.zw;
    r.local = t.xy;
    return r;
}

// A random value per shape, in [0, 1).
fn ff_random(cell: FfCell, salt: f32) -> f32 { return hash21(cell.key, p.seed_ * 13.37 + salt); }

// Flood Fill to Index's size filter: a shape counts when its box diagonal (UV)
// reaches the threshold, and at least a pixel.
fn ff_big_enough(box: vec2f, threshold: f32) -> bool {
    return length(box) >= max(threshold, 1.0 / min(p.size_.x, p.size_.y));
}
