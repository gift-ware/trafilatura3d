// k-means: index of each histogram pixel's nearest centroid.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = texel_small(c).rgb;
    var best = 0;
    var bd = 1e30;
    for (var k = 0; k < p.count; k++) {
        let e = dist2(s, texel_palette(vec2i(k, 0)).rgb);
        if (e < bd) { bd = e; best = k; }
    }
    store_assign(c, gray4(f32(best)));
}
