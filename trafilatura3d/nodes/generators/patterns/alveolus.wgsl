// Lattice point q of row r: rows half a column apart, the first one 3/4 of a column in.
fn alv_point(q: f32, r: f32, cells: vec2f) -> vec2f {
    return vec2f(q + 0.75 - 0.5 * imod2(r), r + 0.5) / cells;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let n = f32(max(p.tiling, 1));
    let cells = vec2f(2.0) * n;
    let g = uv * cells;
    let r0 = floor(g.y);
    // Nearest lattice point, then the second-order neighbours for the cell edges.
    var pts: array<vec2f, 15>;
    var k = 0;
    var best = 0;
    var bd = 1e9;
    for (var dr = -1; dr <= 1; dr++) {
        let r = r0 + f32(dr);
        let q0 = floor(g.x - 0.75 + 0.5 * imod2(r));
        for (var dq = -2; dq <= 2; dq++) {
            let pt = alv_point(q0 + f32(dq), r, cells);
            pts[k] = pt;
            let dd = length(uv - pt);
            if (dd < bd) { bd = dd; best = k; }
            k++;
        }
    }
    let a = pts[best];
    var e = 1e9;                                             // distance to the cell border
    for (var j = 0; j < 15; j++) {
        if (j == best) { continue; }
        let b = pts[j];
        let nn = normalize(b - a);
        e = min(e, dot((a + b) * 0.5 - uv, nn));
    }
    let inr = 0.5 * min(1.0 / cells.x, length(vec2f(0.5 / cells.x, 1.0 / cells.y)));
    let px = pixel_uv();
    var v: f32;
    if (p.gradient_filled_cells == 1) {
        v = clamp(e / inr, 0.0, 1.0) * edge(e, px);
    } else {
        // Smoothstep spheres of radius 3/4 of a column at Interstice Width 0 (tuned), the
        // brightest one winning where they overlap.
        let rad = max(0.75 / cells.x * (1.0 - clamp(p.interstice_width, 0.0, 1.0)), 1e-5);
        let t = clamp(1.0 - bd / rad, 0.0, 1.0);
        v = t * t * (3.0 - 2.0 * t);
    }
    store(c, vec4f(v, v, v, 1.0));
}
