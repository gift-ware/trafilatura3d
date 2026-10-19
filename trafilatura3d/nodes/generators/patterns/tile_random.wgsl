// Tile Random: draws the leaves laid out on the CPU (tile_random.py `layout_texture`), binned on a grid over the
// frame; each pixel blends the leaves of its bin in drawing order. Frame units: 1 = a square of the frame.

fn tr_data(i: i32) -> vec4f {
    let w = size_leaves().x;
    return texel_leaves(vec2i(i % w, i / w));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let g = frame(c, p.non_square_expansion);
    var acc = vec4f(p.background_color.rgb, 1.0);
    if (OUT_OUTPUT_GRAY) { acc = vec4f(vec3f(p.background_color.r), 1.0); }
    if (BACKGROUND_INPUT_CONNECTED) { acc = load_background_input(c); }
    let head = tr_data(0);  // bins x, y, first entry, first leaf
    let bins = vec2i(head.xy);
    let cell = clamp(vec2i(floor(g / frame_size(p.non_square_expansion) * head.xy)), vec2i(0), bins - 1);
    let bin = tr_data(1 + cell.y * bins.x + cell.x);
    let first = i32(bin.x);
    for (var e = first; e < first + i32(bin.y); e++) {
        let row = i32(head.w) + 3 * i32(tr_data(i32(head.z) + e).x);
        let a = tr_data(row);      // centre, signed size (negative = mirrored)
        let b = tr_data(row + 1);  // rotation (turns), pattern specific
        let v = tr_data(row + 2);  // colour (Add/Sub: minus the background colour)
        let q = (rot(-b.x) * (g - a.xy)) / (a.zw * 0.5);
        if (any(abs(q) > vec2f(1.0))) { continue; }
        let w = 2.0 * pixel_uv() / min(abs(a.z), abs(a.w));
        // The pattern per channel and, for a colour image, its tint and coverage.
        var pat = vec4f(0.0);
        var tint = vec3f(1.0);
        if (p.pattern == 0) {
            let t = pat_sample(0, q * 0.5 + 0.5, p.image_input_filtering, w * 0.5);
            if (OUT_OUTPUT_GRAY) { pat = vec4f(t.r); } else { pat = t; tint = t.rgb; }
        } else {
            pat = vec4f(pattern_shape(p.pattern - 1, q, b.y));
        }
        switch p.blending_mode {
            case 0: { acc += pat * v; }
            case 1: { acc = max(acc, pat * v); }
            default: {
                let k = clamp(pat.a * v.a, 0.0, 1.0);
                acc = vec4f(mix(acc.rgb, tint * v.rgb, k), k + acc.a * (1.0 - k));
            }
        }
    }
    acc = clamp(acc, vec4f(0.0), vec4f(1.0));
    if (OUT_OUTPUT_GRAY) { store(c, vec4f(acc.r, acc.r, acc.r, 1.0)); } else { store(c, acc); }
}
