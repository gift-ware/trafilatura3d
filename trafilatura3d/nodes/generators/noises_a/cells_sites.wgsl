// The sites of Cells 1, 3 and 4: one per cell of the Scale grid, at a point spread uniformly over a disc of radius
// SITE_RADIUS cells around the cell's centre (radius SITE_RADIUS x sqrt(random)), its angle turned by Disorder (the
// anisotropy flattens that loop across its angle); `s`: the random stream. SITE_RADIUS fitted on the mean and
// spread of Cells 1 and the walls of Cells 3; REACH is how far a site can be (the patterns' search reach).

const SITE_RADIUS = 1.3;
const REACH = 1.5;

fn site_offset(q: vec2i, per: vec2i, s: f32, dis: f32, aniso: f32, aniso_angle: f32) -> vec2f {
    let h = hw2(q, per, s);
    let rho = sqrt(h.x) * SITE_RADIUS;
    let at_rest = rho * vec2f(cos(TAU * h.y), sin(TAU * h.y));
    let a = TAU * (h.y + dis);
    let moved = rho * vec2f(cos(a), sin(a)) - at_rest;
    let along = rot2(vec2f(1.0, 0.0), aniso_angle);
    let across = vec2f(-along.y, along.x);
    return at_rest + along * dot(moved, along) + across * dot(moved, across) * (1.0 - clamp(aniso, 0.0, 1.0));
}

// Cells 1's map at x (in cells): on every site a pattern (shapes.wgsl `kind`) of half extent `half` cells (capped to
// what the 7x7 search reaches), turned by angle +- angle_random / 2, darkened by up to lum_random, overlaid with Max
// (Hemisphere: lowered instead); `aa`: a pixel in pattern units.
fn cells_domes_at(x: vec2f, per: vec2i, kind: i32, half_in: vec2f, angle: f32, angle_random: f32, lum_random: f32,
              dis: f32, aniso: f32, aniso_angle: f32) -> f32 {
    let half = clamp(half_in, vec2f(1e-4), vec2f(3.0 - REACH));
    let aa = px_cells(f32(per.x)) / min(half.x, half.y);
    let i = vec2i(floor(x));
    var v = 0.0;
    for (var j = -3; j <= 3; j++) {
        for (var ii = -3; ii <= 3; ii++) {
            let q = i + vec2i(ii, j);
            let site = vec2f(q) + 0.5 + site_offset(q, per, p.seed_, dis, aniso, aniso_angle);
            let h = hw2(q, per, p.seed_ + 41.0);
            let local = rot2(x - site, -(angle + (h.x - 0.5) * angle_random)) / half;
            if (max(abs(local.x), abs(local.y)) > 1.0) { continue; }
            let lum = 1.0 - h.y * lum_random;
            if (kind == 16) {
                v = max(v, max(shape(kind, local, 0.0, aa) - (1.0 - lum), 0.0));
            } else {
                v = max(v, shape(kind, local, 0.0, aa) * lum);
            }
        }
    }
    return v;
}
