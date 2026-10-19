// Organisms: Gaussian bells (two per cell, random height, max-blended) on the Scale lattice, warped
// by a gradient noise (Warp Intensity, negative warps the other way). The height shows as a dim body
// plus a thin bright membrane where it crosses 1 (heights reach 1.6), so only the tall bells become
// rings around a darker core.
fn organisms(x: vec2f, per: i32, s: f32) -> f32 {
    let i = vec2i(floor(x));
    let pv = vec2i(per);
    var v = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var xx = -1; xx <= 1; xx++) {
            let q = i + vec2i(xx, y);
            for (var k = 0; k < 2; k++) {
                let sk = s + f32(k) * 7.31;
                let ph = TAU * (p.disorder + hcell(q, pv, sk + 3.7));
                let pt = vec2f(q) + hcell2(q, pv, sk) + 0.1 * vec2f(cos(ph), sin(ph));
                let d = x - pt;
                v = max(v, pow(hcell(q, pv, sk + 9.1), 1.5) * exp(-dot(d, d) / (2.0 * 0.22 * 0.22)));
            }
        }
    }
    return v;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv0 = expand_uv(pix_uv(c), p.non_square_expansion);
    let per = clamp(p.scale, 1, 1024);
    let uv = uv0 + p.warp_intensity * 0.5 / f32(per) * warp_vec(uv0, per, 2, p.seed_ + 5.0);
    let t = 1.6 * organisms(uv * f32(per), per, p.seed_);
    let v = 0.3 * min(t, 1.0) + 0.7 * pow(1.0 - abs(t - 1.0), 12.0);
    store(c, vec4f(v, v, v, 1.0));
}
