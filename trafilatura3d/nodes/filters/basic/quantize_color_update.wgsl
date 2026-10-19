// k-means: each centroid moves to the weighted mean of its pixels; w = their total weight (0: the
// centroid is unused and keeps its place).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let hs = size_small();
    var acc = vec3f(0.0);
    var wsum = 0.0;
    for (var y = 0; y < hs.y; y++) {
        for (var x = 0; x < hs.x; x++) {
            if (i32(texel_assign(vec2i(x, y)).r + 0.5) == c.x) {
                let s = texel_small(vec2i(x, y));
                acc += s.rgb * s.a;
                wsum += s.a;
            }
        }
    }
    let old = texel_palette(c);
    store_palette(c, select(vec4f(old.rgb, 0.0), vec4f(acc / max(wsum, 1e-6), wsum), wsum > 1e-6));
}
