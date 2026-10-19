// Sort the palette (scatter each entry to its rank): used entries first, then by the ID sort mode.
fn spread(v: u32) -> u32 {
    var x = v & 0xffu;
    x = (x | (x << 8u)) & 0x00f00fu;
    x = (x | (x << 4u)) & 0x0c30c3u;
    x = (x | (x << 2u)) & 0x249249u;
    return x;
}
fn sort_key(e: vec4f) -> f32 {
    let rgb = from_space(e.rgb);
    if (p.id_sort_mode == 1) { return rgb_to_hsl(rgb).x; }
    if (p.id_sort_mode == 2) { return -e.a; }
    let q = vec3u(round(rgb * 255.0));
    let z = (spread(q.r) << 2u) | (spread(q.g) << 1u) | spread(q.b);
    return -f32(z);  // from white to black
}
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let e = texel_palette(c);
    let used = e.a > 0.0;
    let key = sort_key(e);
    var rank = 0;
    for (var j = 0; j < p.count; j++) {
        if (j == c.x) { continue; }
        let o = texel_palette(vec2i(j, 0));
        let ou = o.a > 0.0;
        let ok = sort_key(o);
        var before = false;
        if (ou != used) { before = ou; }
        else if (ok != key) { before = ok < key; }
        else { before = j < c.x; }
        if (before) { rank++; }
    }
    store_palette(vec2i(rank, 0), e);
}
