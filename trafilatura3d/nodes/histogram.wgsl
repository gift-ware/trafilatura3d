// One workgroup per row of the output (size() = (WG x rows, WG)): it counts the input rows y = row (mod rows) into
// workgroup memory, then writes the counts of its row (texel x < BINS) and its minimum and maximum (BINS, BINS + 1).

var<workgroup> counts: array<atomic<u32>, 1024>;  // bin * 4 + channel
var<workgroup> lows: array<atomic<u32>, 4>;
var<workgroup> highs: array<atomic<u32>, 4>;

// Floats as u32 in the same order (negative ones too), for atomicMin / atomicMax.
fn ordered(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn unordered(u: u32) -> f32 {
    return bitcast<f32>(select(~u, u & 0x7fffffffu, (u & 0x80000000u) != 0u));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
    for (var i = t; i < BINS * 4u; i += u32(WG * WG)) { atomicStore(&counts[i], 0u); }
    if (t < 4u) {
        atomicStore(&lows[t], 0xffffffffu);
        atomicStore(&highs[t], 0u);
    }
    workgroupBarrier();
    let d = size_image();
    let rows = size().x / WG;
    var lo = vec4f(3.4e38);
    var hi = vec4f(-3.4e38);
    for (var y = i32(wg.x); y < d.y; y += rows) {
        for (var x = i32(t); x < d.x; x += WG * WG) {
            let v = texel_image(vec2i(x, y));
            lo = min(lo, v);
            hi = max(hi, v);
            let b = min(vec4u(clamp(v, vec4f(0.0), vec4f(1.0)) * f32(BINS)), vec4u(BINS - 1u));
            for (var k = 0u; k < 4u; k++) { atomicAdd(&counts[b[k] * 4u + k], 1u); }
        }
    }
    for (var k = 0u; k < 4u; k++) {
        atomicMin(&lows[k], ordered(lo[k]));
        atomicMax(&highs[k], ordered(hi[k]));
    }
    workgroupBarrier();
    let row = i32(wg.x);
    for (var i = t; i < BINS; i += u32(WG * WG)) {
        store(vec2i(i32(i), row), vec4f(f32(atomicLoad(&counts[i * 4u])), f32(atomicLoad(&counts[i * 4u + 1u])),
                                        f32(atomicLoad(&counts[i * 4u + 2u])), f32(atomicLoad(&counts[i * 4u + 3u]))));
    }
    if (t == 0u) {
        store(vec2i(i32(BINS), row), vec4f(unordered(atomicLoad(&lows[0])), unordered(atomicLoad(&lows[1])),
                                           unordered(atomicLoad(&lows[2])), unordered(atomicLoad(&lows[3]))));
        store(vec2i(i32(BINS) + 1, row), vec4f(unordered(atomicLoad(&highs[0])), unordered(atomicLoad(&highs[1])),
                                               unordered(atomicLoad(&highs[2])), unordered(atomicLoad(&highs[3]))));
    }
}
