// VARIANT 0: long, nearly vertical strands (Fur 1); 1: short horizontal strands following waves
// (Fur 2); 2: short bristles fanning out from fan centres (Fur 3). Strands are straight segments, one
// per cell of a periodic grid, overlaid with Max. Scale repeats the strands: their
// length and width are given at the default Scale and shrink with the grid (thinner than a pixel
// they fade, averaged over it); the fans and waves stay across the tile (Waves controls).

// Slope (dx per unit up) of Fur 3's strands at x, at the top of the tile: n fans across the tile,
// upright at each fan's centre, leaning outwards towards its edges.
fn fur_fan(x: f32, n: f32) -> f32 {
    if (n < 1.0) { return 0.0; }
    return 3.5 * (fract(x * n) - 0.5);
}

// Direction (unit, image y down) of a strand centred at q (tile units); h: its random number;
// k, t: the variant's wave controls (Fur 2: waves across the tile and twist; Fur 3: fans).
fn fur_dir(q: vec2f, h: f32, k: f32, t: f32) -> vec2f {
    if (VARIANT == 0) {
        let a = (h - 0.5) * 0.09 * TAU;
        return vec2f(sin(a), -cos(a));
    }
    if (VARIANT == 1) {
        // Turned by two crossing waves (t: Waves Rotation, the amplitude in half turns; k: Waves Scale, from 2
        // waves across the tile at 0 to 1 at 1), their phases from the seed.
        let w = TAU * (2.0 - clamp(k, 0.0, 1.0));
        let ph = fract(vec2f(0.618034, 0.754878) * (p.seed_ + 1.0)) * TAU;
        let a = TAU * 0.5 * t * (sin(w * q.x + ph.x) + sin(w * q.y + ph.y));
        return normalize(vec2f(cos(a), sin(a) + (h - 0.5) * 0.1));
    }
    // The fans spread most at the top of the tile, least in its lower half (periodic: it tiles).
    let s = mix(fur_fan(q.x, floor(k)), fur_fan(q.x, floor(k) + 1.0), fract(k)) * (0.7 + 0.3 * cos(TAU * q.y));
    return normalize(vec2f(s + (h - 0.5) * 0.4, -1.0));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let q = uv_of(c) * asp;
    var dis = 0.0;
    var k = 0.0;
    var t = 0.0;
    /*@params*/
    // Per variant: cells per Scale unit, strand length (tile units), width (pixels at 256),
    // luminance range, largest |dir.x| and |dir.y| (for the neighbourhood searched).
    var cells = vec2f(40.0, 1.0);
    var len = 0.36;
    var wid = 1.3;
    var lum = vec2f(0.25, 1.0);
    var ext = vec2f(0.3, 1.0);
    if (VARIANT == 1) { cells = vec2f(5.0, 40.0); len = 0.12; wid = 0.8; lum = vec2f(0.1, 0.7); ext = vec2f(1.0, 0.4); }
    if (VARIANT == 2) { cells = vec2f(18.0, 18.0); len = 0.07; wid = 1.2; lum = vec2f(0.4, 1.0); ext = vec2f(0.9, 1.0); }
    let z = DEFAULT_SCALE / f32(max(p.scale, 1));
    len *= z;
    wid *= z;
    let per = max(vec2i(round(cells * f32(max(p.scale, 1)) * asp)), vec2i(1));
    let cell = asp / vec2f(per);
    let x = q / cell;
    let i = vec2i(floor(x));
    let reach = vec2i(ceil(0.5 * len * ext / cell)) + 1;
    // Edge softness: a pixel of 512 at the default Scale, scaled with the pattern, the same at every resolution
    // (the look of the preview at 512 everywhere; strands thinner than a pixel break up and alias, not fade).
    let aa = asp.x / 512.0 * z;
    let half_w = 0.5 * wid / 256.0;
    var v = 0.0;
    for (var j = -reach.y; j <= reach.y; j++) {
        for (var ii = -reach.x; ii <= reach.x; ii++) {
            let qc = i + vec2i(ii, j);
            let h = hw2(qc, per, p.seed_);
            let g = hw2(qc, per, p.seed_ + 3.0);
            var ctr = (vec2f(qc) + h) * cell;
            let dir = fur_dir(ctr, g.x, k, t);
            let l = len * (0.6 + 0.4 * g.y);
            ctr += dir * l * dis;  // Disorder slides the strands along themselves
            let dd = q - ctr;
            let along = dot(dd, dir);
            let s = along / l + 0.5;
            if (s <= 0.0 || s >= 1.0) { continue; }
            let across = length(dd - dir * along);
            let cover = clamp((half_w + 0.5 * aa - across) / aa, 0.0, 1.0);
            let b = mix(lum.x, lum.y, hw(qc, per, p.seed_ + 5.0)) * sqrt(sin(3.14159265 * s));
            v = max(v, b * cover);
        }
    }
    store(c, vec4f(v, v, v, 1.0));
}
