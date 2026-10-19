// Farthest-point sampling, centroid k: the weighted mean for k = 0, else the histogram pixel
// farthest from the centroids so far. Other entries are copied.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let hs = size_small();
    if (c.x < p.k) { store_palette(c, texel_palette(c)); return; }
    if (c.x > p.k) { store_palette(c, vec4f(0.0)); return; }
    var best = vec4f(0.0);
    if (p.k == 0) {
        var acc = vec3f(0.0);
        var wsum = 0.0;
        for (var y = 0; y < hs.y; y++) {
            for (var x = 0; x < hs.x; x++) {
                let s = texel_small(vec2i(x, y));
                acc += s.rgb * s.a;
                wsum += s.a;
            }
        }
        best = vec4f(acc / max(wsum, 1e-6), 1.0);
    } else {
        var far = -1.0;
        for (var y = 0; y < hs.y; y++) {
            for (var x = 0; x < hs.x; x++) {
                let s = texel_small(vec2i(x, y));
                let m = texel_mind(vec2i(x, y)).r * s.a;
                if (m > far) { far = m; best = vec4f(s.rgb, 1.0); }
            }
        }
    }
    store_palette(c, best);
}
