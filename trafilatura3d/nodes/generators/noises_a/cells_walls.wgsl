// Cells 3 (ours): the walls where the domes of Cells 1 meet, found analytically. Of the sites of Cells 1
// (cells_sites.wgsl) the two nearest, s1 and s2: the wall is their bisector, e the distance to it, and two domes
// 1 - (d / R)^2 of radius R meet there with a slope jump of 2 |s2 - s1| / R^2. The wall's darkness is that jump x
// (1 - e / w), w a pixel of 512 (the preview's: the same walls at every resolution), divided by C3_SOFT x
// (1 - Hardness)^C3_HARD (Hardness darkens and widens them), white elsewhere (Invert: black). Both fitted on the
// darkness and wall coverage.

const C3_SOFT = 1.9;
const C3_HARD = 0.6;  // Hardness acts as (1 - Hardness)^C3_HARD

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let per = vec2i(round(f32(max(p.scale, 1)) * asp));
    let k = vec2f(per) / asp;
    let x = (uv_of(c) * asp + p.tile_offset) * k;
    let dis = p.disorder * p.disorder_speed;
    let i = vec2i(floor(x));
    var d1 = 1e9;
    var d2 = 1e9;
    var s1 = vec2f(0.0);
    var s2 = vec2f(0.0);
    for (var j = -3; j <= 3; j++) {
        for (var ii = -3; ii <= 3; ii++) {
            let q = i + vec2i(ii, j);
            let s = vec2f(q) + 0.5 + site_offset(q, per, p.seed_, dis, p.disorder_anisotropy, p.disorder_anisotropy_angle);
            let dd = dot(x - s, x - s);
            if (dd < d1) { d2 = d1; s2 = s1; d1 = dd; s1 = s; } else if (dd < d2) { d2 = dd; s2 = s; }
        }
    }
    let gap = max(length(s2 - s1), 1e-4);
    let e = (d2 - d1) / (2.0 * gap);  // distance to the bisector, in cells
    let r = 1.5 * max(0.5 * (p.pattern_size.x + p.pattern_size.y) * p.pattern_scale, 1e-3);  // the domes' radius
    let jump = 2.0 * gap / (r * r);
    let w = (asp.x / 512.0) * k.x;  // a pixel of 512, in cells
    let dark = clamp(jump * max(1.0 - e / w, 0.0) / (C3_SOFT * pow(max(1.0 - p.hardness, 1e-3), C3_HARD)), 0.0, 1.0);
    var v = 1.0 - dark;
    if (p.invert != 0) { v = 1.0 - v; }
    store(c, vec4f(v, v, v, 1.0));
}
