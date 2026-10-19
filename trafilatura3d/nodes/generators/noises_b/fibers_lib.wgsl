// Straight fibers scattered on a periodic lattice: per cell `n` segments (half length `half` cells,
// +-40% random, half width `w` cells), direction Angle +- Angle random / 2 (turns), brightness
// 1 - `lum` x random. `tube`: each pixel shows the topmost fiber (random depth) shaded as a cylinder,
// so crossings get a dark rim; otherwise the fibers are ribbons of `lines` parallel threads (thinner
// with a higher `sharp`), screen-blended.
// Neighbourhood of `r` cells around: 1.4 half + 0.25 (disorder) must stay under r.
fn fibers(uv: vec2f, per: i32, r: i32, n: i32, half: f32, w: f32, lines: f32, sharp: f32, lum: f32,
          tube: bool, s: f32) -> f32 {
    let x = uv * f32(per);
    let i = vec2i(floor(x));
    let pv = vec2i(per);
    var v = 0.0;
    var top = -1.0;
    for (var y = -r; y <= r; y++) {
        for (var xx = -r; xx <= r; xx++) {
            let q = i + vec2i(xx, y);
            for (var k = 0; k < n; k++) {
                let sk = s + f32(k) * 2.93;
                let h2 = hcell2(q, pv, sk + 1.7);
                let c = vec2f(q) + hcell2(q, pv, sk) + disorder_disp(h2, p.disorder, p.disorder_speed,
                                                                    p.disorder_anisotropy, p.disorder_anisotropy_angle);
                let h3 = hcell2(q, pv, sk + 4.1);
                let l = rot2(x - c, -TAU * (p.angle + (h3.x - 0.5) * p.angle_random));
                let u = l.x / (half * (0.6 + 0.8 * h3.y));
                if (abs(u) >= 1.0) { continue; }
                let end = 1.0 - pow(u * u, 4.0);  // tubes: rounded ends, ribbons: faded ends
                let across = l.y / (w * select(1.0, sqrt(end), tube));
                if (abs(across) >= 1.0) { continue; }
                let hb = hcell2(q, pv, sk + 7.7);
                let bright = 1.0 - lum * hb.y;
                if (tube) {
                    if (hb.x > top) {
                        top = hb.x;
                        v = bright * (0.7 + 0.3 * top) * sqrt(1.0 - across * across);
                    }
                } else {
                    let threads = pow(0.5 + 0.5 * cos(PI * lines * across), sharp);
                    v = 1.0 - (1.0 - v) * (1.0 - bright * threads * (1.0 - across * across) * end);
                }
            }
        }
    }
    return v;
}
