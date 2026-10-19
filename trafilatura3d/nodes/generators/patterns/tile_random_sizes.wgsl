// Tile Random 2: a grid of moved vertices, cells pinwheeling around them, one bevelled shape per cell.

fn t2_n() -> vec2i { return max(vec2i(p.amount_x, p.amount_y), vec2i(1)); }

// Signed random value of a vertex (Uniform, or Blue Noise: R2 sequence).
fn t2_vert(v: vec2i, salt: u32, dist: i32) -> f32 {
    if (dist == 1) { return fract(rnd1(1, salt) + f32(v.x) * 0.7548777 + f32(v.y) * 0.5698403) * 2.0 - 1.0; }
    return srnd(v, salt);
}

// Vertex (i, j) of the grid (unwrapped), in cells: xy = its position, moved by the random sizes,
// zw = the slopes of its horizontal (dy/dx) and vertical (dx/dy) lines, from the random slant.
fn t2_vertex(i: i32, j: i32) -> vec4f {
    let n = t2_n();
    let w = vec2i(wrapi(i, n.x), wrapi(j, n.y));
    let vuv = (vec2f(w) + 0.5) / vec2f(n);
    let sm = mix(1.0, sample_random_size_map(vuv).r, p.random_size_input_map_multiplier);
    let lm = mix(1.0, sample_random_slant_map(vuv).r, p.random_slant_input_map_multiplier);
    let sz = 0.5 * clamp(p.random_size_multiplier * vec2f(p.random_size_x, p.random_size_y) * sm, vec2f(0.0), vec2f(1.0));
    let sl = 0.3 * clamp(p.random_slant_multiplier * vec2f(p.random_slant_x, p.random_slant_y) * lm, vec2f(0.0), vec2f(1.0));
    let o = sz * vec2f(t2_vert(w, 11u, p.random_size_distribution), t2_vert(w, 12u, p.random_size_distribution));
    let k = sl * vec2f(t2_vert(w, 13u, p.random_slant_distribution), t2_vert(w, 14u, p.random_slant_distribution));
    return vec4f(vec2f(f32(i), f32(j)) + clamp(o, vec2f(-0.48), vec2f(0.48)), k);
}

// Corner where the vertical line of vertex a meets the horizontal line of vertex b.
fn t2_meet(a: vec4f, b: vec4f) -> vec2f {
    // x = a.x + a.w (y - a.y), y = b.y + b.z (x - b.x)
    let x = (a.x + a.w * (b.y - b.z * b.x - a.y)) / max(1.0 - a.w * b.z, 1e-3);
    return vec2f(x, b.y + b.z * (x - b.x));
}

// Corners (top left, top right, bottom right, bottom left) of cell (i, j). The cells pinwheel around
// the vertices: each side lies on the line of one of the four vertices, alternating with the
// parity of the cell, so neighbouring cells share their sides and meet in T junctions.
fn t2_cell(i: i32, j: i32) -> array<vec2f, 4> {
    let n = t2_n();
    let v00 = t2_vertex(i, j);
    let v10 = t2_vertex(i + 1, j);
    let v11 = t2_vertex(i + 1, j + 1);
    let v01 = t2_vertex(i, j + 1);
    var l = v00; var t = v10; var r = v11; var b = v01;
    if (((wrapi(i, n.x) + wrapi(j, n.y)) & 1) == 1) { l = v01; t = v00; r = v10; b = v11; }
    return array<vec2f, 4>(t2_meet(l, t), t2_meet(r, t), t2_meet(r, b), t2_meet(l, b));
}

fn t2_cross(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

fn t2_corner(c: array<vec2f, 4>, k: i32) -> vec2f {
    switch wrapi(k, 4) { case 0: { return c[0]; } case 1: { return c[1]; } case 2: { return c[2]; } default: { return c[3]; } }
}
// Point at perimeter parameter u (corner k at u = k).
fn t2_perimeter(c: array<vec2f, 4>, u: f32) -> vec2f {
    let k = i32(floor(u));
    return mix(t2_corner(c, k), t2_corner(c, k + 1), u - floor(u));
}
fn t2_seg(p0: vec2f, a: vec2f, b: vec2f) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p0 - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(p0 - a - ab * t);
}

// A mask range hides the shapes whose value lies in it (start/end in either order; an empty range hides
// none, the default 0 / 0); Invert hides the others instead.
fn t2_visible(v: f32, a: f32, b: f32, invert: i32) -> bool {
    let hidden = max(a, b) > min(a, b) && v >= min(a, b) && v <= max(a, b);
    return hidden == (invert == 1);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / p.size_;
    let n = t2_n();
    let nf = vec2f(n);
    let g = uv * nf;
    // The cell containing the pixel.
    var cg: array<vec2f, 4>;
    var ci = vec2i(0);
    var best = -1e9;  // the cell the pixel is most inside of
    let g0 = vec2i(floor(g));
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            let q = t2_cell(g0.x + di, g0.y + dj);
            var score = 1e9;
            for (var k = 0; k < 4; k++) {
                let ed = t2_corner(q, k + 1) - t2_corner(q, k);
                score = min(score, t2_cross(ed, g - t2_corner(q, k)) / max(length(ed), 1e-6));
            }
            if (score > best) { best = score; ci = g0 + vec2i(di, dj); cg = q; }
        }
    }
    let cell = vec2i(wrapi(ci.x, n.x), wrapi(ci.y, n.y));
    var cov = 0.0;
    var v = 0.0;
    var luv = vec2f(0.0);
    let cs = min(1.0 / nf.x, 1.0 / nf.y);
    // Masks.
    let area = clamp(0.5 * abs(t2_cross(cg[2] - cg[0], cg[3] - cg[1])) / 8.0, 0.0, 1.0);
    var cen_cell = (cg[0] + cg[1] + cg[2] + cg[3]) * 0.25 / nf;
    let r0 = rnd(cell, 900u);
    let m = sample_mask_map(cen_cell).r;
    let vis_r = t2_visible(r0, p.mask_random_start, p.mask_random_end, p.mask_random_invert);
    let vis_a = t2_visible(area, p.mask_by_cell_area_start, p.mask_by_cell_area_end, p.mask_by_cell_area_invert);
    let vis_m = t2_visible(m, p.mask_map_input_start, p.mask_map_input_end, p.mask_map_input_invert);
    if (vis_r && vis_a && vis_m) {
        // Cell corners in uv; the shape slides its corners along the border (rotation).
        let cu = array<vec2f, 4>(cg[0] / nf, cg[1] / nf, cg[2] / nf, cg[3] / nf);
        let rot_f = p.rotation + p.rotation_random * srnd(cell, 910u);
        var s: array<vec2f, 4>;
        for (var k = 0; k < 4; k++) {
            let f = rot_f + p.corners_position_random * 0.3 * srnd(cell, 911u + u32(k));
            s[k] = t2_perimeter(cu, f32(k) + f - 4.0 * floor((f32(k) + f) / 4.0));
        }
        // Scale about the centre, random position in the freed space.
        let cen = (s[0] + s[1] + s[2] + s[3]) * 0.25;
        let sc = p.scale_multiplier * (1.0 - p.scale_random * rnd(cell, 920u));
        let bmin0 = min(min(s[0], s[1]), min(s[2], s[3]));
        let bmax0 = max(max(s[0], s[1]), max(s[2], s[3]));
        let off = vec2f(srnd(cell, 921u), srnd(cell, 922u)) * p.random_position_multiplier * (1.0 - sc) * (bmax0 - bmin0) * 0.5;
        for (var k = 0; k < 4; k++) { s[k] = cen + off + (s[k] - cen) * sc; }
        let bmin = min(min(s[0], s[1]), min(s[2], s[3]));
        let bmax = max(max(s[0], s[1]), max(s[2], s[3]));
        luv = clamp((uv - bmin) / max(bmax - bmin, vec2f(1e-6)), vec2f(0.0), vec2f(1.0));
        // Erosion: half the interstice + the corner radius (the radius is added back after).
        let per_axis = p.enable_per_axis_bevel_control == 1;
        let rm = mix(1.0, sample_round_corners_radius_map(cen_cell).r, p.round_corners_radius_input_map_mult);
        let gap = p.interstice_thickness * cs * 0.5;
        var nrm: array<vec2f, 4>;
        var cc: array<f32, 4>;
        var inr = 1e9;  // distance of the centre to the eroded sides: caps the corner radius
        for (var k = 0; k < 4; k++) {
            let dd = normalize(t2_corner(s, k + 1) - t2_corner(s, k) + vec2f(1e-9));
            nrm[k] = vec2f(-dd.y, dd.x);
            inr = min(inr, dot(nrm[k], cen + off - s[k]) - gap);
        }
        let radius = select(min(p.round_corners_radius * rm * cs * 0.5, max(inr * 0.9, 0.0)), 0.0, per_axis);
        let e = gap + radius;
        for (var k = 0; k < 4; k++) {
            cc[k] = dot(nrm[k], s[k]) + e;
        }
        var ev: array<vec2f, 4>;
        for (var k = 0; k < 4; k++) {
            let a = nrm[wrapi(k - 1, 4)];
            let b = nrm[k];
            let ca = cc[wrapi(k - 1, 4)];
            let det = a.x * b.y - a.y * b.x;
            ev[k] = vec2f(ca * b.y - cc[k] * a.y, a.x * cc[k] - b.x * ca) / select(det, 1e-9, abs(det) < 1e-9);
        }
        var valid = true;
        for (var k = 0; k < 4; k++) {
            if (t2_cross(t2_corner(ev, k + 1) - t2_corner(ev, k), t2_corner(ev, k + 2) - t2_corner(ev, k + 1)) <= 0.0) { valid = false; }
        }
        if (valid) {
            var din = 1e9;
            var dout = 1e9;
            var lx = 1e9;
            var ly = 1e9;
            for (var k = 0; k < 4; k++) {
                let ld = dot(nrm[k], uv) - cc[k];
                din = min(din, ld);
                dout = min(dout, t2_seg(uv, t2_corner(ev, k), t2_corner(ev, k + 1)));
                if ((k & 1) == 0) { ly = min(ly, ld); } else { lx = min(lx, ld); }
            }
            let sd = select(-dout, din, din >= 0.0) + radius;
            cov = edge(sd, pixel_uv());
            // Bevel.
            var dcen = 1e9;
            for (var k = 0; k < 4; k++) { dcen = min(dcen, dot(nrm[k], cen + off) - cc[k]); }
            var refd = cs * 0.5;
            if (p.bevel_distance_mode == 1) { refd = max(dcen + radius, 0.0); }
            if (p.bevel_distance_mode == 2) { refd = 0.25; }
            let bm = mix(1.0, load_bevel_distance_map(c).r, p.bevel_distance_input_map_mult);
            var t = 1.0;
            if (per_axis) {
                let bx = p.bevel_distance_x * refd * bm;
                let by = p.bevel_distance_y * refd * bm;
                t = min(select(1.0, clamp(lx / bx, 0.0, 1.0), bx > 1e-6), select(1.0, clamp(ly / by, 0.0, 1.0), by > 1e-6));
            } else {
                let b = p.bevel_distance_multiplier * refd * bm;
                if (b > 1e-6) { t = clamp(sd / b, 0.0, 1.0); }
            }
            v = cov * mix(t, sqrt(max(t * (2.0 - t), 0.0)), p.bevel_rounded_curve);
        }
    }
    store_output(c, vec4f(v, v, v, 1.0));
    store_uv(c, vec4f(luv * step(0.5, cov), rnd(cell, 950u), cov));
}
