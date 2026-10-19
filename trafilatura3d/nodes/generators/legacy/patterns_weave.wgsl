// Weave Twill (Legacy), ours: two families of thread floats on a lattice of 16 x 16 cells per tile (Tiling tiles
// across). Every cell holds one float of each family: an arc along the thread, (1 - u^2)^a with u in [-1, 1] over
// its length, bent sideways by `bend` u^2 and rounded across, (1 - v^2)^b. A family is (direction cos/sin, half
// length, half width in cells) in `shape*`, (a, b, bend, gain) in `profile*` and the float's place in its cell
// (x, y in 0-1; z: how far, in cells, the float reaches) in `place*`. Disorder moves each float by `disorder` cells
// at a random angle (per cell and family, the same in every tile: the `moves` table, patterns_weave_moves.wgsl). The
// floats add up over `lift`, then go through a tone curve (`tone*`) fitted on the legacy node's
// value distribution. `rotate` samples the weave turned 45 degrees and enlarged by sqrt 2 (it still tiles
// without Disorder).

// One float of a family at x - centre = e.
fn float_at(e: vec2f, shape: vec4f, profile: vec4f) -> f32 {
    let u = dot(e, shape.xy) / shape.z;
    let v = (dot(e, vec2f(-shape.y, shape.x)) - profile.z * u * u) / shape.w;
    if (abs(u) >= 1.0 || abs(v) >= 1.0) { return 0.0; }
    return profile.w * pow(1.0 - u * u, profile.x) * pow(1.0 - v * v, profile.y);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var uv = (vec2f(c) + 0.5) / vec2f(d);
    if (p.rotate != 0) {
        let q = uv - 0.5;
        uv = vec2f(0.5 * q.x + 0.5 * q.y, -0.5 * q.x + 0.5 * q.y) + 0.5;
    }
    let x = fract(uv * f32(p.tiles)) * 16.0;
    // The cells whose floats of either family can reach x (both places are within the cell).
    let r1 = p.place1.z + p.disorder;
    let r2 = p.place2.z + p.disorder;
    let lo = vec2i(floor(min(x - p.place1.xy - r1, x - p.place2.xy - r2)));
    let hi = vec2i(floor(max(x - p.place1.xy + r1, x - p.place2.xy + r2)));
    var cols: array<i32, 16>;  // the wrapped columns, once (integer modulo is slow)
    let w = min(hi.x - lo.x, 15);
    for (var i = 0; i <= w; i++) { cols[i] = wrap1(lo.x + i, 16); }
    var s = p.lift;
    for (var j = lo.y; j <= hi.y; j++) {
        let row = wrap1(j, 16);
        // On this row, the cells whose unmoved float centre is within its reach of x (a disc per family).
        let dy1 = x.y - f32(j) - p.place1.y;
        let dy2 = x.y - f32(j) - p.place2.y;
        let h1 = sqrt(max(r1 * r1 - dy1 * dy1, 0.0));
        let h2 = sqrt(max(r2 * r2 - dy2 * dy2, 0.0));
        let i0 = max(i32(ceil(min(x.x - p.place1.x - h1, x.x - p.place2.x - h2))) - lo.x, 0);
        let i1 = min(i32(floor(max(x.x - p.place1.x + h1, x.x - p.place2.x + h2))) - lo.x, w);
        for (var i = i0; i <= i1; i++) {
            let cell = vec2f(f32(lo.x + i), f32(j));
            var e1 = x - cell - p.place1.xy;
            var e2 = x - cell - p.place2.xy;
            if (p.disorder > 0.0) {
                let m = textureLoad(src_moves, vec2i(cols[i], row), 0);
                e1 -= m.xy;
                e2 -= m.zw;
            }
            if (dot(e1, e1) < p.place1.z * p.place1.z) { s += float_at(e1, p.shape1, p.profile1); }
            if (dot(e2, e2) < p.place2.z * p.place2.z) { s += float_at(e2, p.shape2, p.profile2); }
        }
    }
    // The tone curve: 12 values at sums -0.2, 0, 0.2 ... 2.0, linearly interpolated (gradual towards white).
    let lut = array<f32, 12>(p.tone0.x, p.tone0.y, p.tone0.z, p.tone0.w, p.tone1.x, p.tone1.y, p.tone1.z, p.tone1.w,
                             p.tone2.x, p.tone2.y, p.tone2.z, p.tone2.w);
    let f = clamp((s + 0.2) / 0.2, 0.0, 11.0);
    let k = min(i32(floor(f)), 10);
    let v = mix(lut[k], lut[k + 1], f - f32(k));
    store(c, vec4f(v, v, v, 1.0));
}
