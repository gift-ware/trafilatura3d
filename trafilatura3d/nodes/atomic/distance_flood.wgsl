// Distance between pixels, wrapping around on tiled axes.
fn delta(a: vec2f, b: vec2f) -> vec2f {
    var v = abs(a - b);
    let d = vec2f(size());
    if (tile_h()) { v.x = min(v.x, d.x - v.x); }
    if (tile_v()) { v.y = min(v.y, d.y - v.y); }
    return v;
}

fn metric(v: vec2f) -> f32 {
    if (p.metric == 1) { return v.x + v.y; }
    if (p.metric == 2) { return max(v.x, v.y); }
    return length(v);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var best = seed_at(c);
    var best_d = select(1e20, metric(delta(best, vec2f(c))), best.x >= 0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let q = c + vec2i(i, j) * p.step;
            let inside_x = tile_h() || (q.x >= 0 && q.x < d.x);
            let inside_y = tile_v() || (q.y >= 0 && q.y < d.y);
            if (!(inside_x && inside_y)) { continue; }
            let s = seed_at(q);
            if (s.x < 0.0) { continue; }
            let dd = metric(delta(s, vec2f(c)));
            if (dd < best_d) { best_d = dd; best = s; }
        }
    }
    store_seeds(c, jfa_pack(vec4f(best.x, 0.0, 0.0, best.y)));
}
