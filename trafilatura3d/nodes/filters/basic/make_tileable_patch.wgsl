@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let n = 1 << u32(clamp(p.octave, 0, 6));
    let g = uv * f32(n);
    let cell0 = vec2i(floor(g));
    let seed = pcg(bitcast<u32>(p.seed_));
    let wobble = (fbm(uv, vec2i(n * 4), p.seed_ + 5.0) - 0.5) * p.mask_warping / 100.0;
    // The mask: a bell (1 - 4 r^2)^5 over the patch (r from its centre, 0 at its edge), through Levels
    // 1 - Size .. 1 - Size x Precision.
    let lo = 1.0 - p.mask_size;
    let hi = 1.0 - p.mask_size * p.mask_precision;
    var prio: array<f32, 25>;
    var mask: array<f32, 25>;
    var col: array<vec4f, 25>;
    for (var j = 0; j < 5; j++) {
        for (var i = 0; i < 5; i++) {
            let k = j * 5 + i;
            let cell = cell0 + vec2i(i - 2, j - 2);
            let key = wrap(cell, vec2i(n));
            // The cell's random values: 16-bit halves of a PCG chain of five words.
            let w0 = pcg(u32(key.x) ^ pcg(u32(key.y) ^ seed));
            let w1 = pcg(w0);
            let w2 = pcg(w1);
            let w3 = pcg(w2);
            let w4 = pcg(w3);
            let h = vec2f(f32(w0 & 0xFFFFu), f32(w0 >> 16u)) / 65535.0;
            let h2 = vec2f(f32(w1 & 0xFFFFu), f32(w1 >> 16u)) / 65535.0;
            let h3 = vec2f(f32(w2 & 0xFFFFu), f32(w2 >> 16u)) / 65535.0;
            let hw = vec2f(f32(w3 & 0xFFFFu), f32(w3 >> 16u)) / 65535.0;
            let hh = vec2f(f32(w4 & 0xFFFFu), f32(w4 >> 16u)) / 65535.0;
            // Size: Pattern size (% of a cell) +- Size Variation, each axis on its own.
            let pct = vec2f(p.pattern_size_width + (hw.x - hw.y) * p.size_variation,
                            p.pattern_size_height + (hh.x - hh.y) * p.size_variation);
            let cells = max(pct / 100.0, vec2f(1e-3));
            let centre = vec2f(cell) + 0.5 + p.disorder * vec2f(cos(6.2831853 * h.x), sin(6.2831853 * h.x));
            let turns = (p.rotation + h2.y * p.rotation_variation) / 360.0;
            let pc = rot(g - centre, -turns) / cells;  // in the patch, -0.5 .. 0.5
            let b = max(1.0 - 4.0 * dot(pc, pc), 0.0);
            let bell = b * b * b * b * b;
            let warped = bell - wobble;
            if (hi - lo < 1e-5) { mask[k] = select(0.0, 1.0, warped > lo); }
            else { mask[k] = clamp((warped - lo) / (hi - lo), 0.0, 1.0); }
            if (any(abs(pc) > vec2f(0.5))) { mask[k] = 0.0; }
            var v = sample_input(pc + 0.5);
            // Color / Luminosity Variation: each channel darkened by a random part of it.
            if (INPUT_GRAY) { v = gray4(v.r * (1.0 - h3.x * p.luminosity_variation)); }
            else {
                v = vec4f(v.rgb * (1.0 - vec3f(h3.x, h3.y, h2.x) * p.color_variation), v.a);
            }
            col[k] = v;
            prio[k] = fract(h.y * 13.37 + hh.x * 7.91 + 0.123);  // the stacking order
        }
    }
    var o = p.background_color;
    if (INPUT_GRAY) { o = gray4(o.r); }
    // Composite the stamps from the lowest priority up.
    var last = -1.0;
    for (var it = 0; it < 25; it++) {
        var best = -1;
        var bp = 2.0;
        for (var k = 0; k < 25; k++) {
            if (prio[k] > last && prio[k] < bp) { bp = prio[k]; best = k; }
        }
        if (best < 0) { break; }
        last = bp;
        o = mix(o, col[best], mask[best]);
    }
    store(c, o);
}
