// Quantize, split step: the clusters with the largest sum of squared distances (as many as free slots allow) are
// split in two along their axis of largest variance, at +- one standard deviation; the new halves take the slots
// after the valid ones (valid slots stay contiguous from 0). Clusters of one color are never split.
fn split_offset(k: i32) -> vec3f {
    let v = texel_palette(vec2i(k, 1)).xyz;
    if (v.x >= v.y && v.x >= v.z) { return vec3f(sqrt(v.x), 0.0, 0.0); }
    if (v.y >= v.z) { return vec3f(0.0, sqrt(v.y), 0.0); }
    return vec3f(0.0, 0.0, sqrt(v.z));
}
fn splittable(k: i32) -> bool { return texel_palette(vec2i(k, 0)).w >= 0.0 && texel_palette(vec2i(k, 1)).w > 1e-7; }
// Rank of a splittable cluster among the splittable ones, by decreasing error (ties: lower index first).
fn split_rank(k: i32) -> i32 {
    let e = texel_palette(vec2i(k, 1)).w;
    var r = 0;
    for (var j = 0; j < p.k_; j++) {
        let f = texel_palette(vec2i(j, 1)).w;
        if (j != k && splittable(j) && (f > e || (f == e && j < k))) { r++; }
    }
    return r;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    var n = 0;
    var cand = 0;
    for (var j = 0; j < p.k_; j++) {
        if (texel_palette(vec2i(j, 0)).w >= 0.0) { n++; }
        if (splittable(j)) { cand++; }
    }
    let m = min(cand, p.k_ - n);
    let k = c.x;
    var e = texel_palette(vec2i(k, 0));
    if (e.w >= 0.0) {
        if (splittable(k) && split_rank(k) < m) { e = vec4f(e.xyz - split_offset(k), e.w); }
    } else if (k - n < m) {
        for (var i = 0; i < p.k_; i++) {
            if (splittable(i) && split_rank(i) == k - n) {
                e = vec4f(texel_palette(vec2i(i, 0)).xyz + split_offset(i), 0.0);
            }
        }
    }
    store_palette(vec2i(k, 0), e);
    store_palette(vec2i(k, 1), texel_palette(vec2i(k, 1)));
}
