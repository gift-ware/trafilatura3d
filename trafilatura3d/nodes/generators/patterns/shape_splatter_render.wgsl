// Shape Splatter, pass 2: the tiles gathered per pixel; the highest one wins.

var<private> g_bgs: f32;      // background, smoothed for conforming
var<private> g_best: f32;     // highest tile height so far
var<private> g_uv: vec2f;
var<private> g_id: f32;
var<private> g_rand: f32;
var<private> g_rot: f32;
var<private> g_base: f32;
var<private> g_idx: f32;

fn tg_line_shift(line: i32) -> f32 { return 0.0; }

fn tg_tile(cell: vec2i, base: vec2f) -> Tile {
    let a = texel_tiles_a(cell);
    let b = texel_tiles_b(cell);
    var t: Tile;
    t.visible = a.w >= 0.0;
    t.center = a.xy;
    t.inv = mat2x2f(1.0 / b.x, 0.0, 0.0, 1.0 / b.y) * rot(-a.z);
    t.kind = i32(a.w);
    t.spec = a.z;
    t.color = b;
    t.extra = texel_tiles_c(cell);
    t.key = texel_tiles_d(cell).y;
    return t;
}

fn ss_pattern(k: i32, q: vec2f, w: f32) -> f32 {
    if (!ss_connected(k)) { return shape(2, q, 0.0, w); }
    return pat_sample(k, q * 0.5 + 0.5, 0, w * 0.5).r;
}

fn tg_eval(t: Tile, q: vec2f, d: vec2f) -> vec4f {
    if (any(abs(q) > vec2f(1.0))) { return vec4f(0.0); }   // a tile claims its square only
    let v = ss_pattern(t.kind, q, tg_aa(t));
    if (p.cutout_black_pixels == 1 && v <= 0.0) { return vec4f(0.0); }
    let rigid = t.color.w + p.skew_from_bg_slope * dot(t.extra.zw, d);
    let conformed = g_bgs + (t.color.w - t.extra.y);
    let th = mix(rigid, conformed, p.conform_to_background) + v * t.extra.x;
    if (th > g_best) {
        g_best = th;
        g_uv = q * 0.5 + 0.5;
        g_id = f32(t.kind + 1);
        g_rand = t.color.z;
        g_rot = fract(t.spec);
        g_base = t.color.w;
        g_idx = t.key;
    }
    return vec4f(0.0);
}

fn tg_fold(acc: vec4f, v: vec4f) -> vec4f { return acc; }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv0 = (vec2f(c) + 0.5) / p.size_;
    let bg = load_background_height(c).r;
    g_bgs = bg;
    let rad = p.smooth_conformed_background * 8.0 / 256.0;
    if (rad > 0.0) {
        var acc = 0.0;
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) { acc += sample_background_height(uv0 + vec2f(f32(i), f32(j)) * rad).r; }
        }
        g_bgs = acc / 9.0;
    }
    g_best = -1e30;
    g_id = 0.0;
    let n = ss_counts();
    _ = tg_render(uv0 - ss_global_offset(), n, false, tg_reach(ss_extent()), false, false, vec4f(0.0));
    var has = g_id > 0.0;
    var h = bg;
    if (has) {
        if (p.flatten_pattern_base == 1) { h = g_best; } else if (g_best >= bg) { h = g_best; } else { has = false; }
    }
    store_height(c, vec4f(h, h, h, 1.0));
    if (has) {
        // Data 2 G (scale) comes from the tile texture by index.
        let cell = vec2i(i32(g_idx) % n.x, i32(g_idx) / n.x);
        let sc = texel_tiles_d(cell).x;
        store_splatter_data_1(c, vec4f(g_uv, g_id, g_rand));
        store_splatter_data_2(c, vec4f(g_rot, sc, g_base, g_idx / f32(n.x * n.y)));
    } else {
        store_splatter_data_1(c, vec4f(0.0));
        store_splatter_data_2(c, vec4f(0.0));
    }
}
