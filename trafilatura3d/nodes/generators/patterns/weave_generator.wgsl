// One direction of the twill: blocks two cells long, each holding `w` parallel stitches. `g` is in
// cells with x along the blocks, `start` the cell where the block of this row begins (mod 4), `px`
// one pixel in cells and `gh` the half gaps (along, across). Returns (stitch height, stitch coverage, gap thread height).
fn wg_layer(g: vec2f, start: f32, px: vec2f, w: f32, gh: vec2f) -> vec3f {
    // Along: the block spans [start, start + 2] cells and reaches gh.x into the gaps at both ends.
    var s = g.x - start;
    s -= 4.0 * floor(s / 4.0);
    if (s > 2.0 + gh.x) { s -= 4.0; }
    let along = min(s + gh.x, 2.0 + gh.x - s);                       // to the nearest end
    // Across: w stitches per cell, gh.y of gap on each side of every stitch.
    let tw = fract(g.y) * w;
    let across = min(fract(tw), 1.0 - fract(tw)) / w - gh.y;
    // Ends sink with a quadratic profile over half a cell, Shape = how deep (at the default:
    // 1 - 0.5 (1 - d / 0.5)^2, gaps and threads tuned on it too).
    let e = 1.0 - min(along / 0.5, 1.0);
    let h = 1.0 - p.shape * e * e;
    let cov = edge(along, px.x) * edge(across, px.y);
    // Between two parallel stitches, away from the cell corners: a half-height thread, rounded across.
    let q = (across + gh.y) / max(2.0 * gh.y, 1e-6);
    let interior = abs(g.x - round(g.x)) - gh.x;
    let thread = 0.5 * (1.0 - q * q) * edge(along, px.x) * edge(-across, px.y) * edge(interior, px.x);
    return vec3f(h, cov, thread);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = vec2f(f32(max(p.tile_x, 1)), f32(max(p.tile_y, 1)));
    let g = frame(c, p.non_square_expansion) * t;
    let px = pixel_uv() * t;
    let w = f32(max(p.weave, 1));
    let gh = 0.5 * clamp(p.gap, vec2f(0.0), vec2f(0.98)) / w;
    // 2/2 twill: in row r the horizontal block starts at cell r - 1, in column c the vertical one at
    // row c - 2 (mod 4), each covering two cells.
    let hl = wg_layer(g, floor(g.y) - 1.0, px, w, gh);
    let vl = wg_layer(g.yx, floor(g.x) - 2.0, px.yx, w, gh.yx);
    let stitch = select(vl.x, hl.x, hl.y >= vl.y);
    let cov = max(hl.y, vl.y);
    let v = mix(max(hl.z, vl.z), stitch, cov);
    let m1 = select(0.0, hl.y, hl.y >= vl.y);
    let m2 = select(vl.y, 0.0, hl.y >= vl.y);
    store_height(c, vec4f(v, v, v, 1.0));
    store_opacity(c, vec4f(cov, cov, cov, 1.0));
    store_mask_1(c, vec4f(m1, m1, m1, 1.0));
    store_mask_2(c, vec4f(m2, m2, m2, 1.0));
}
