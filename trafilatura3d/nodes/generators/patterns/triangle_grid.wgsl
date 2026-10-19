fn tri_counts() -> vec2i { return max(vec2i(p.x_amount, p.y_amount), vec2i(1)); }

// Random value of the colour source (global seed, or the manual one).
fn tri_crnd(cell: vec2i, salt: u32) -> f32 {
    if (p.random_color_seed_mode == 1) {
        let h = pcg(u32(cell.x) * 1597334677u ^ pcg(u32(cell.y) * 3812015801u ^ pcg(salt ^ u32(p.random_color_seed) * 2654435761u)));
        return f32(h >> 8u) / 16777216.0;
    }
    return rnd(cell, salt);
}

// Vertex of the unwrapped grid point (i, j): position in uv, z.
fn tri_vertex(i: i32, j: i32) -> vec3f {
    let n = tri_counts();
    let w = vec2i(wrapi(i, n.x), wrapi(j, n.y));
    let buv = (vec2f(w) + 0.5) / vec2f(n);
    var off = (0.5 - vec2f(rnd(w, 1u), rnd(w, 2u))) * p.random_position * p.random_position_multiplier
        + p.vector_map_displacement * (0.5 - sample_vector_map(buv).rg);
    // Rotation, plus a random turn of up to +- half a turn x Rotation Disorder.
    off = rot(p.rotation + (rnd(w, 3u) - 0.5) * p.rotation_disorder) * off;
    // Quincunx: alternate rows (x) / columns (y) shifted by +-half the offset, then at most a cell's diagonal.
    off += (0.5 - vec2f(f32(w.y & 1), f32(w.x & 1))) * vec2f(p.quincunx_offset_x, p.quincunx_offset_y);
    let l = length(off);
    off = select(off / max(l, 1e-8) * min(l, 1.41421356), vec2f(0.0), l < 0.005);  // tiny offsets: none
    let z = max(mix(0.5, sample_height(buv).r, p.height_input_multiplier) + (rnd(w, 4u) - 0.5) * p.height_random,
                0.001);
    return vec3f((vec2f(f32(i), f32(j)) + 0.5 + off) / vec2f(n), z);
}

// Colour source of a vertex.
fn tri_vsrc(i: i32, j: i32, v: vec3f) -> vec3f {
    let n = tri_counts();
    let w = vec2i(wrapi(i, n.x), wrapi(j, n.y));
    switch p.color_source {
        case 0: { return vec3f(v.z); }
        case 1: { return sample_color_input(fract(v.xy)).rgb; }
        case 2: { return vec3f(tri_crnd(w, 10u)); }
        default: { return vec3f(tri_crnd(w, 10u), tri_crnd(w, 11u), tri_crnd(w, 12u)); }
    }
}

fn tri_nz(a: vec3f, b: vec3f, c: vec3f) -> f32 {
    let n = tri_counts();
    let s = vec3f(vec2f(n), 1.0);
    let m = cross((b - a) * s, (c - a) * s);
    return abs(m.z) / max(length(m), 1e-8);
}

// Distance from x to the segment a-b.
fn tri_seg(x: vec2f, a: vec2f, b: vec2f) -> f32 {
    let e = b - a;
    let t = clamp(dot(x - a, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
    return length(x - a - t * e);
}

// The overlap blending of the chosen output (0 Min, 1 Max, 2 Depth Test, 3 Alpha Blend).
fn tri_mode() -> i32 {
    switch p.color_output {
        case 0, 1: { return select(1, 2, p.blend_mode == 1); }
        case 2: { return p.thin_line_blend_mode + 1; }
        case 3: { return select(0, 2, p.distance_blend_mode == 1); }
        default: { return 2; }
    }
}

// Signed distance from x to the edge a->b of a triangle (positive inside, given the winding sign).
fn tri_edge(x: vec2f, a: vec2f, b: vec2f, sgn: f32) -> f32 {
    let e = b - a;
    return sgn * (e.x * (x.y - a.y) - e.y * (x.x - a.x)) / max(length(e), 1e-8);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = tri_counts();
    let uv = frame(c, p.non_square_expansion);
    let g = uv * vec2f(n) - 0.5;   // vertices sit at cell centres
    let px = pixel_uv();
    let reach = clamp(1 + i32(ceil(max(p.random_position.x, p.random_position.y) * p.random_position_multiplier * 0.5
        + max(p.quincunx_offset_x, p.quincunx_offset_y) * 0.5 + p.vector_map_displacement * 0.5)), 1, 3);
    let bg = p.background_color.rgb;
    var acc = vec3f(0.0);
    var best = -1e30;
    var found = false;
    let mode = tri_mode();
    if (mode == 0) { acc = vec3f(1.0); found = true; }   // start: 1 in uv, past any thickness
    if (mode == 1) { acc = vec3f(-1e30); }
    if (mode == 3) { acc = bg; found = true; }
    let g0 = vec2i(floor(g));
    for (var dj = -reach; dj <= reach; dj++) {
        for (var di = -reach; di <= reach; di++) {
            let ci = g0.x + di;
            let cj = g0.y + dj;
            let cw = vec2i(wrapi(ci, n.x), wrapi(cj, n.y));
            let va = tri_vertex(ci, cj);
            let vb = tri_vertex(ci + 1, cj);
            let vc = tri_vertex(ci, cj + 1);
            let vd = tri_vertex(ci + 1, cj + 1);
            // Diagonal: false = a-d (-45 degrees), true = b-c (45 degrees).
            var diag_bc = false;
            switch p.triangulation {
                case 1: { diag_bc = true; }
                case 2: { diag_bc = false; }
                case 3: { diag_bc = (cw.y & 1) == 1; }
                case 4: { diag_bc = (cw.x & 1) == 1; }
                default: {
                    let ad = min(tri_nz(va, vb, vd), tri_nz(va, vd, vc));
                    let bc = min(tri_nz(va, vb, vc), tri_nz(vb, vd, vc));
                    diag_bc = bc > ad;
                }
            }
            let sa = tri_vsrc(ci, cj, va);
            let sb = tri_vsrc(ci + 1, cj, vb);
            let sc = tri_vsrc(ci, cj + 1, vc);
            let sd = tri_vsrc(ci + 1, cj + 1, vd);
            for (var t = 0; t < 2; t++) {
                var p0 = va; var p1 = vb; var p2 = vd;
                var s0 = sa; var s1 = sb; var s2 = sd;
                if (diag_bc) {
                    if (t == 0) { p2 = vc; s2 = sc; } else { p0 = vb; p1 = vd; p2 = vc; s0 = sb; s1 = sd; s2 = sc; }
                } else if (t == 1) { p1 = vd; p2 = vc; s1 = sd; s2 = sc; }
                let e1 = p1.xy - p0.xy;
                let e2 = p2.xy - p0.xy;
                let area2 = e1.x * e2.y - e1.y * e2.x;
                if (abs(area2) < 1e-12) { continue; }
                let sgn = sign(area2);
                let d0 = tri_edge(uv, p0.xy, p1.xy, sgn);
                let d1 = tri_edge(uv, p1.xy, p2.xy, sgn);
                let d2 = tri_edge(uv, p2.xy, p0.xy, sgn);
                let dmin = min(d0, min(d1, d2));
                if (mode == 0) {
                    // Min distance: the nearest edge of every triangle, inside or not.
                    let e = min(tri_seg(uv, p0.xy, p1.xy), min(tri_seg(uv, p1.xy, p2.xy), tri_seg(uv, p2.xy, p0.xy)));
                    acc = min(acc, vec3f(e / max(p.distance_to_edge_thickness, 1e-8)));
                    continue;
                }
                let cov = clamp(dmin / px + 0.5, 0.0, 1.0);
                if ((mode == 3 && cov <= 0.0) || (mode != 3 && dmin < -1e-7)) { continue; }
                // Barycentrics (weights of p0, p1, p2).
                let x = uv - p0.xy;
                let b1 = (x.x * e2.y - x.y * e2.x) / area2;
                let b2 = (e1.x * x.y - e1.y * x.x) / area2;
                let bw = vec3f(1.0 - b1 - b2, b1, b2);
                let z = dot(bw, vec3f(p0.z, p1.z, p2.z));
                let cen = (p0.xy + p1.xy + p2.xy) / 3.0;
                // Triangle colour source.
                var tsrc: vec3f;
                switch p.color_source {
                    case 0: { tsrc = vec3f((p0.z + p1.z + p2.z) / 3.0); }
                    case 1: { tsrc = sample_color_input(fract(cen)).rgb; }
                    case 2: { tsrc = vec3f(tri_crnd(vec2i(cw.x * 2 + t, cw.y), 20u)); }
                    default: {
                        let tc = vec2i(cw.x * 2 + t, cw.y);
                        tsrc = vec3f(tri_crnd(tc, 20u), tri_crnd(tc, 21u), tri_crnd(tc, 22u));
                    }
                }
                var v: vec3f;
                switch p.color_output {
                    case 0: { v = bw.x * s0 + bw.y * s1 + bw.z * s2; }
                    case 1: { v = tsrc; }
                    case 2: {
                        let hw = 0.5 / 256.0;
                        let line = clamp((hw - dmin) / px + 0.5, 0.0, 1.0);
                        v = mix(bg, mix(p.line_color.rgb, tsrc, p.color_source_opacity), line);
                    }
                    case 3: { v = vec3f(dmin / max(p.distance_to_edge_thickness, 1e-8)); }
                    default: { v = vec3f(3.0 * min(bw.x, min(bw.y, bw.z))); }
                }
                switch mode {
                    case 0: { acc = min(acc, v); }
                    case 1: { acc = max(acc, v); }
                    case 2: { if (z > best) { best = z; acc = v; } }
                    default: { acc = mix(acc, v, cov); }
                }
                found = true;
            }
        }
    }
    if (!found) { acc = bg; }
    store(c, vec4f(clamp(acc, vec3f(0.0), vec3f(1.0)), 1.0));
}
