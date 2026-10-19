// Tile-grid engine (Tile Generator, Tile Sampler, Splatter, Shape Splatter): a grid of counts.x by
// counts.y cells, one tile per cell, gathered per pixel from the cells around it (`reach` cells in
// every direction, enough for scaled, rotated and displaced tiles). Lines (rows, or columns when
// `vert`) can be shifted. The node defines:
//   fn tg_tile(cell: vec2i, base: vec2f) -> Tile    the tile of a (wrapped) cell, base = its centre in uv
//   fn tg_eval(t: Tile, q: vec2f, d: vec2f) -> vec4f   its value at local q (d = uv offset): q spans
//                                                   [-1,1]^2 plus half an antialiasing width
//   fn tg_fold(acc: vec4f, v: vec4f) -> vec4f       blends a tile value onto the accumulator
//   fn tg_line_shift(line: i32) -> f32              shift of a (wrapped) line, in cells
// Tiles are folded in render order (Tile.key, later on top) when `ordered`, else as found. After tile.wgsl (Tile,
// tg_aa).

const TG_MAX = 32;

fn tg_render(uv: vec2f, counts: vec2i, vert: bool, reach: i32, ordered: bool, reverse: bool, init: vec4f) -> vec4f {
    let n = max(counts, vec2i(1));
    let g = uv * vec2f(n);
    let gs = select(g, g.yx, vert);
    let ns = select(n, n.yx, vert);
    var acc = init;
    var keys: array<f32, TG_MAX>;
    var vals: array<vec4f, TG_MAX>;
    var count = 0;
    let r = clamp(reach, 1, 6);
    for (var dj = -r; dj <= r; dj++) {
        let j = i32(floor(gs.y)) + dj;
        let jw = wrapi(j, ns.y);
        let sh = tg_line_shift(jw);
        for (var di = -r; di <= r; di++) {
            let i = i32(floor(gs.x - sh)) + di;
            let iw = wrapi(i, ns.x);
            let cell = select(vec2i(iw, jw), vec2i(jw, iw), vert);
            // Base position of the unwrapped cell (each periodic image of a tile is its own cell).
            let bs = vec2f(f32(i) + 0.5 + sh, f32(j) + 0.5);
            let base = select(bs, bs.yx, vert) / vec2f(n);
            let t = tg_tile(cell, base);
            if (!t.visible) { continue; }
            let d = uv - base - t.center;
            let q = t.inv * d;
            // Half an antialiasing width past the border: the tile's edge pixels.
            if (any(abs(q) > vec2f(1.0 + 0.5 * tg_aa(t)))) { continue; }
            let v = tg_eval(t, q, d);
            if (!ordered) {
                acc = tg_fold(acc, v);
            } else if (count < TG_MAX) {
                keys[count] = select(t.key, -t.key, reverse);
                vals[count] = v;
                count++;
            }
        }
    }
    if (ordered) {
        // Insertion sort by render order, then fold back to front.
        for (var a = 1; a < count; a++) {
            let k = keys[a];
            let v = vals[a];
            var b = a;
            loop {
                if (b == 0) { break; }
                if (keys[b - 1] <= k) { break; }
                keys[b] = keys[b - 1];
                vals[b] = vals[b - 1];
                b--;
            }
            keys[b] = k;
            vals[b] = v;
        }
        for (var a = 0; a < count; a++) { acc = tg_fold(acc, vals[a]); }
    }
    return acc;
}

// Reach (cells to gather around a pixel) for tiles whose half extent, in cells, is up to `ext`.
fn tg_reach(ext: f32) -> i32 { return i32(ceil(ext + 0.5)); }
