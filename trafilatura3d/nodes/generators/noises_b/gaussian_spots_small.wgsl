// Small sharp spots: positive Gaussian bells (sigma 0.13
// cell, two per cell, random weight) on four lattices from Scale, doubling; each finer lattice 2.5 x
// brighter, so the finest spots dominate over a faint haze of the coarser ones. Levels put the
// background at black.
fn bells(uv: vec2f, per: i32, s: f32, dis: vec4f) -> f32 {
    let x = uv * f32(per);
    let i = vec2i(floor(x));
    let pv = vec2i(per);
    var v = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var xx = -1; xx <= 1; xx++) {
            let q = i + vec2i(xx, y);
            for (var k = 0; k < 2; k++) {
                let sk = s + f32(k) * 7.31;
                let pt = vec2f(q) + hcell2(q, pv, sk) + disorder_disp(hcell2(q, pv, sk + 3.7), dis.x, dis.y, dis.z, dis.w);
                let d = x - pt;
                v += hcell(q, pv, sk + 9.1) * exp(-dot(d, d) / (2.0 * 0.13 * 0.13));
            }
        }
    }
    return v;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion) + p.tile_offset;
    let dis = vec4f(p.disorder, p.disorder_speed, p.disorder_anisotropy, p.disorder_anisotropy_angle);
    var per = max(p.scale, 1);
    var n = 0.0;
    var a = 1.0 / 15.625;
    for (var o = 0; o < 4; o++) {
        n += a * bells(uv, per, p.seed_ + f32(o) * 31.0, dis);
        a *= 2.5;
        per *= 2;
    }
    let v = sat(0.65 * (n - 0.039));
    store(c, vec4f(v, v, v, 1.0));
}
