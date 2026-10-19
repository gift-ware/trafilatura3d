// The Tile Sampler's first pass: each cell's tile computed once, into three records (r0, r1, r2) at the cell.
// The cells are numbered in the lines' space (columns when the lines are vertical, `ts_vert`), over the whole
// frame (Non Square Expansion: the grid repeats along the longer axis, each repeat its own tiles: they read the
// maps where they are). Records: r0 = tile centre in the frame (less Global Offset), inverse transform's first
// column; r1 = its second column, Pattern Specific, pattern (TS_HIDDEN: not drawn); r2 = colour and opacity.

const TS_HIDDEN = 100000.0;

fn ts_gray() -> bool { return p.gray_ == 1; }

// The records' grid: the cells of the lines' space over the frame.
fn ts_records() -> vec2i {
    let n = ts_counts();
    let fs = frame_size(p.non_square_expansion);
    let ns = select(n, n.yx, ts_vert());
    return ns * vec2i(round(select(fs, fs.yx, ts_vert())));
}

// One invocation per output texel (the dispatch's size is the output's: frame() and the pixel sizes read it),
// each computing the cells it lands on every output size apart (more cells than texels in a small output).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let m = ts_records();
    let d = size();
    if (any(vec2i(id.xy) >= d)) { return; }
    for (var y = i32(id.y); y < m.y; y += d.y) {
        for (var x = i32(id.x); x < m.x; x += d.x) {
            ts_cell(vec2i(x, y));
        }
    }
}

fn ts_cell(k: vec2i) {
    let n = ts_counts();
    let vert = ts_vert();
    let ns = select(n, n.yx, vert);
    // As the tile grid engine places a cell (tiles.wgsl, tg_render): line j shifted, the cell wrapped.
    let jw = wrapi(k.y, ns.y);
    let sh = tg_line_shift(jw);
    let iw = wrapi(k.x, ns.x);
    let cell = select(vec2i(iw, jw), vec2i(jw, iw), vert);
    let bs = vec2f(f32(k.x) + 0.5 + sh, f32(k.y) + 0.5);
    let base = select(bs, bs.yx, vert) / vec2f(n);
    let t = tg_tile(cell, base);
    let b = base + t.center;
    store_r0(k, vec4f(b, t.inv[0]));
    store_r1(k, vec4f(t.inv[1], t.spec, select(TS_HIDDEN, f32(t.kind), t.visible)));
    store_r2(k, t.color);
}
