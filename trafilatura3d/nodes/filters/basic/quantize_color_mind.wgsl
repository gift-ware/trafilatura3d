// Squared distance of each histogram pixel to its nearest centroid so far (adds centroid k).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = texel_small(c);
    var m = dist2(s.rgb, texel_palette(vec2i(p.k, 0)).rgb);
    let prev = texel_mind(c).r;
    if (p.k > 0) { m = min(m, prev); }
    store_mind(c, gray4(m));
}
