// Arc Pavement: columns 1 cell wide (Scale per image); in each column, arcs of a circle of radius
// 1/sqrt(2) centred below the column axis, stacked vertically every 1/(Arcs Amount + 1) cell, so the
// arcs of two neighbouring columns meet at 45 degrees on the column seam (where the band joints are
// at multiples of the band height). Bricks split each 90-degree arc by angle: even bands (from the
// top) Pattern Amount - 1 bricks with a joint on the axis, odd bands Pattern Amount bricks centred
// on it (the staggering of the rows); the joints are radial.

const AP_R = 0.70710678;

// Joint j of a band, moved by Global Pattern Width Random (in brick units).
fn ap_joint(fid: vec2i, j: i32) -> f32 {
    return f32(j) + p.global_pattern_width_random * 0.45 * srnd(fid + vec2i(j + 64, 0), 2u);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = max(p.scale, 1);
    let u = frame(c, p.non_square_expansion) * f32(s);
    let col = i32(round(u.x));
    let dx = u.x - f32(col);
    let nb = p.arcs_amount + 1;
    let a = f32(nb);
    // Band coordinate: 0 on the seam at the cell top, the arcs rising towards the column axis.
    let k = i32(floor((u.y + sqrt(AP_R * AP_R - dx * dx) - 0.5) * a));
    let kw = wrapi(k, nb * s);
    let fid = vec2i(wrapi(col, s) * 128, kw);
    // Bricks on this band, randomised down to the minimum.
    let odd = (kw & 1) == 1;
    let base = max(1, p.pattern_amount - select(1, 0, odd));
    let rn = i32(round(f32(base) * (1.0 - p.pattern_amount_random * rnd(fid, 1u))));
    let nf = f32(max(rn, min(base, p.pattern_minimum_amount)));
    // Angle about the centre of the band's middle arc (joints perpendicular to the arcs).
    let phi = atan2(dx, (f32(k) + 0.5) / a + 0.5 - u.y) / (0.5 * PI);
    let off = select(0.0, 0.5, odd);
    let t = phi * nf + off;
    var m = i32(floor(t));
    var b0 = ap_joint(fid, m);
    var b1 = ap_joint(fid, m + 1);
    if (t < b0) {
        m -= 1;
        b1 = b0;
        b0 = ap_joint(fid, m);
    } else if (t > b1) {
        m += 1;
        b0 = b1;
        b1 = ap_joint(fid, m + 1);
    }
    let bid = fid + vec2i(m + 64, 0);
    // The brick is a rectangle on the middle arc, turned with it: as long as the arc between its two
    // joints and as thick as the band across the arc.
    let pc = 0.5 * (b0 + b1) - off;
    let psi = pc / nf * 0.5 * PI;
    let cs = vec2f(cos(psi), sin(psi));
    let dp = u - vec2f(f32(col) + AP_R * cs.y, (f32(k) + 0.5) / a + 0.5 - AP_R * cs.x);
    let half = vec2f(AP_R * 0.25 * PI * (b1 - b0) / nf, 0.5 / a * cs.x);
    let e = 2.0 * pc / nf;                                   // -1 .. 1 between the ends of the arc
    let hdec = 1.0 - p.pattern_height_decrease * e * e;
    let sx = max(p.pattern_scale * p.pattern_width * (1.0 - p.pattern_width_random * rnd(bid, 3u)), 1e-4);
    let sy = max(p.pattern_scale * p.pattern_height * (1.0 - p.pattern_height_random * rnd(bid, 4u)) * hdec, 1e-4);
    let q = vec2f(dot(dp, cs), dot(dp, vec2f(-cs.y, cs.x))) / (half * vec2f(sx, sy));
    var v = vec4f(0.0, 0.0, 0.0, 1.0);
    if (all(abs(q) <= vec2f(1.0))) {
        // Brick size in uv, for the image input's filtering.
        let w = pixel_uv() * f32(s) / max(min(half.x * sx, half.y * sy), 1e-6);
        let lum = 1.0 - p.color_random * rnd(bid, 5u);
        if (p.pattern == 0) {
            let px = pat_sample(0, q * 0.5 + 0.5, p.input_image_filtering, w * 0.5);
            v = vec4f(px.rgb * lum, 1.0);
        } else {
            let sh = pattern_shape(p.pattern - 1, q, p.pattern_variation) * lum;
            v = vec4f(sh, sh, sh, 1.0);
        }
    }
    if (OUT_OUTPUT_GRAY) { store(c, vec4f(v.r, v.r, v.r, 1.0)); } else { store(c, v); }
}
