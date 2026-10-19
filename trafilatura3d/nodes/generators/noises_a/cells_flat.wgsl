// Cells 4: a flat value on the cells around the sites of Cells 1 (cells_sites.wgsl): a pixel
// per site, then a Distance fills each site's Voronoi cell.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let per = vec2i(round(f32(max(p.scale, 1)) * asp));
    let x = uv_of(c) * vec2f(per);
    let i = vec2i(floor(x));
    let dis = p.disorder * p.disorder_speed;
    var best = 1e9;
    var cell = i;
    var site = vec2f(0.0);
    for (var j = -3; j <= 3; j++) {
        for (var ii = -3; ii <= 3; ii++) {
            let q = i + vec2i(ii, j);
            let s = vec2f(q) + 0.5 + site_offset(q, per, p.seed_, dis, 0.0, 0.0);
            let e = s - x;
            let dd = dot(e, e);
            if (dd < best) { best = dd; cell = q; site = s; }
        }
    }
    let id_ = wrap(cell, max(per, vec2i(1)));
    var v = hash21(id_, p.seed_ + 77.0);
    if (p.color_source == 1) { v = hash21(id_, f32(p.pseudorandom_seed) + 0.5); }
    else if (p.color_source == 2) {
        // The input at the cell's site (wrapped into the image).
        v = sample_input(fract(site / vec2f(per))).r;
    }
    store(c, vec4f(v, v, v, 1.0));
}
