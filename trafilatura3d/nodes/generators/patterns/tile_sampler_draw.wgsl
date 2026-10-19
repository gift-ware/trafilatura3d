// The Tile Sampler's second pass: each tile of the records (tile_sampler_cells.wgsl) drawn as a quad, with its
// copies a frame period away up to `copies_` periods on each side (the image tiles), over the background in the
// target; the blending is the target's (Add/Sub adds, Max keeps the larger, Alpha Blend in the tiles' order).
// Instances in the tiles' render order (cell index, Reverse Rendering Order flips it), each tile's repeats (Non
// Square Expansion) and copies one after the other. A pixel is drawn by the tiles whose square, plus half an
// antialiasing width, holds its centre: the value at the centre as the gather of the tile grid engine gives it.

const TS_HIDDEN = 100000.0;

fn ts_gray() -> bool { return p.gray_ == 1; }

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) centre: vec2f,
    @location(1) @interpolate(flat) inv: vec4f,
    @location(2) @interpolate(flat) color: vec4f,
    @location(3) @interpolate(flat) spec_kind: vec2f,
};

fn ts_inv(v: vec4f) -> mat2x2f { return mat2x2f(v.xy, v.zw); }

// Premultiplied contribution (rgb) and coverage (a) of a tile at local q. Built-in shapes have hard edges (a pixel
// is in or out by its centre); Pattern Inputs keep their filtering.
fn ts_eval(t: Tile, q: vec2f) -> vec4f {
    let op = t.color.a;
    if (t.kind >= 0) {
        let s = pattern_shape(t.kind, q, t.spec);
        return vec4f(s * op * t.color.rgb, s * op);
    }
    let s = pat_sample(-t.kind - 1, q * 0.5 + 0.5, p.pattern_input_filtering, tg_aa(t) * 0.5);
    if (ts_gray()) { return vec4f(op * s.rgb * t.color.rgb, s.r * op); }
    return vec4f(op * s.a * s.rgb * t.color.rgb, s.a * op);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var o: VOut;
    o.pos = vec4f(2.0, 2.0, 2.0, 1.0);  // outside the clip volume: not drawn
    let n = ts_counts();
    let vert = ts_vert();
    let fs = frame_size(p.non_square_expansion);
    let reps = vec2i(round(fs));
    let span = 2 * vec2i(p.copies_) + 1;
    let ncopies = span.x * span.y;
    let nreps = reps.x * reps.y;
    let i = i32(ii);
    let copy = i % ncopies;
    let rep = (i / ncopies) % nreps;
    var idx = i / (ncopies * nreps);
    if (p.reverse_rendering_order == 1) { idx = n.x * n.y - 1 - idx; }
    // The record: the cell (idx row by row), in the lines' space, in its repeat of the frame.
    let cell = vec2i(idx % n.x, idx / n.x) + vec2i(rep % reps.x, rep / reps.x) * n;
    let k = select(cell, cell.yx, vert);
    let r0 = textureLoad(src_rec0, k, 0);
    let r1 = textureLoad(src_rec1, k, 0);
    if (r1.w >= TS_HIDDEN) { return o; }
    // The centre brought into the frame, then this copy's period.
    let period = vec2f(fs);
    let shift = (vec2f(f32(copy % span.x), f32(copy / span.x)) - p.copies_) * period;
    let centre = r0.xy - period * floor(r0.xy / period) + shift;
    let inv = mat2x2f(r0.zw, r1.xy);
    var t: Tile;
    t.inv = inv;
    // The square plus half an antialiasing width (and a little more: every pixel centre the test can keep).
    let e = 1.0 + 0.5 * tg_aa(t) + 1e-3;
    let det = determinant(inv);
    if (abs(det) < 1e-12) { return o; }
    let m = mat2x2f(inv[1].y, -inv[0].y, -inv[1].x, inv[0].x) * (1.0 / det);  // q -> uv offset
    let reach = abs(m[0]) * e + abs(m[1]) * e;  // half extent of the quad in the frame
    let at = centre + p.global_offset / vec2f(n);  // in the frame (frame())
    if (any(at + reach <= vec2f(0.0)) || any(at - reach >= period)) { return o; }
    let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                                  vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
    let uv = at + m * (corners[vi] * e);
    // frame(c) = (c + 0.5) / scale: pixel units, then clip space (y down).
    let scale = select(p.size_, vec2f(min(p.size_.x, p.size_.y)), p.non_square_expansion == 1);
    let px = uv * scale / p.size_;
    o.pos = vec4f(px.x * 2.0 - 1.0, 1.0 - px.y * 2.0, 0.0, 1.0);
    o.centre = centre;
    o.inv = vec4f(inv[0], inv[1]);
    o.color = textureLoad(src_rec2, k, 0);
    o.spec_kind = r1.zw;
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let c = vec2i(floor(in.pos.xy));
    let n = vec2f(ts_counts());
    let uv = frame(c, p.non_square_expansion) - p.global_offset / n;  // Global Offset in cells
    var t: Tile;
    t.inv = ts_inv(in.inv);
    t.color = in.color;
    t.spec = in.spec_kind.x;
    t.kind = i32(in.spec_kind.y);
    let q = t.inv * (uv - in.centre);
    if (any(abs(q) > vec2f(1.0 + 0.5 * tg_aa(t)))) { discard; }
    let v = ts_eval(t, q);
    if (p.blending_mode == 2) { return v; }  // Alpha Blend: premultiplied over
    return vec4f(v.rgb, 0.0);                // Add/Sub, Max: the background's alpha stays
}
