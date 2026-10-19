// The Splatter's tiles drawn as quads over the background in the layer, blended by Max (order-free). Instance =
// cell (row by row) x copy: the copies a whole image apart, from p.copy_lo_ for p.copy_n_ images along each axis,
// as many as can reach the window Zoom and Pan show (splatter.py). A pixel takes a tile whose square, plus half an
// antialiasing width, holds its centre.

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) at: vec2f,
    @location(1) @interpolate(flat) inv: vec4f,
    @location(2) @interpolate(flat) color: vec4f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var o: VOut;
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);  // outside the clip volume: not drawn
    let n = sp_n();
    let span = vec2i(p.copy_n_);
    let copy = i32(ii) % (span.x * span.y);
    let idx = i32(ii) / (span.x * span.y);
    let cell = vec2i(idx % n, idx / n);
    let shift = p.copy_lo_ + vec2f(f32(copy % span.x), f32(copy / span.x));
    let t = sp_tile(cell);
    let at = (vec2f(cell) + 0.5) / f32(n) + shift + t.center;  // in the pattern's uv
    let e = 1.0 + 0.5 * tg_aa(t) + 1e-3;
    let det = determinant(t.inv);
    if (abs(det) < 1e-12) { return o; }
    let m = mat2x2f(t.inv[1].y, -t.inv[0].y, -t.inv[1].x, t.inv[0].x) * (1.0 / det);  // q -> uv offset
    let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                                  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
    // The pattern's uv to the image's (sp_uv inverted), then clip space (y down).
    let pan = vec2f(p.pan_x, p.pan_y) / 100.0;
    let reach = (abs(m[0]) + abs(m[1])) * e * sp_magnify();
    let mid = (at + pan - 0.5) * sp_magnify() + 0.5;
    if (any(mid + reach <= vec2f(0.0)) || any(mid - reach >= vec2f(1.0))) { return o; }
    let px = (at + m * (corners[vi] * e) + pan - 0.5) * sp_magnify() + 0.5;
    o.pos = vec4f(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0, 0.0, 1.0);
    o.at = at;
    o.inv = vec4f(t.inv[0], t.inv[1]);
    o.color = t.color;
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    var t: Tile;
    t.inv = mat2x2f(in.inv.xy, in.inv.zw);
    let q = t.inv * (sp_uv(vec2i(floor(in.pos.xy))) - in.at);
    if (any(abs(q) > vec2f(1.0 + 0.5 * tg_aa(t)))) { discard; }
    let s = pat_sample(0, q * 0.5 + 0.5, p.image_input_filtering, tg_aa(t) * 0.5);
    return vec4f(s.rgb * in.color.rgb, 0.0);  // the background's alpha stays
}
