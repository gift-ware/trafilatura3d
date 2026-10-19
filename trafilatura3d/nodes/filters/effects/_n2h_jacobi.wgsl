// One Jacobi step of the slope integration: each pixel becomes the mean of its neighbours' heights
// plus the rise from them to it (the slope halfway, times one pixel). Without tiling the border has
// no neighbour outside (Neumann condition).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let e = 1.0 / vec2f(d);
    let gp = texel_g(c).xy;
    var acc = 0.0;
    var n = 0.0;
    let offs = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
    for (var k = 0; k < 4; k++) {
        let q = c + offs[k];
        if (!(tile_h() || (q.x >= 0 && q.x < d.x)) || !(tile_v() || (q.y >= 0 && q.y < d.y))) { continue; }
        let gm = (gp + texel_g(q).xy) * 0.5;
        acc += texel_h(q).r - dot(gm, vec2f(offs[k]) * e);
        n += 1.0;
    }
    store_out(c, fx_gray(acc / max(n, 1.0)));
}
