// Quantize, palette pass 2: one cluster (the mean of the grid), the other slots invalid.
// Palette texture (k_ x 2): row 0 = center (xyz) and pixel count (w; -1 = invalid slot), row 1 = per-axis variance
// (xyz) and sum of squared distances (w).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    var center = vec4f(0.0, 0.0, 0.0, -1.0);
    if (c.x == 0) {
        let n = size_down();
        var acc = vec3f(0.0);
        for (var y = 0; y < n.y; y++) { for (var x = 0; x < n.x; x++) { acc += texel_down(vec2i(x, y)).xyz; } }
        center = vec4f(acc / f32(n.x * n.y), 0.0);
    }
    store_palette(vec2i(c.x, 0), center);
    store_palette(vec2i(c.x, 1), vec4f(0.0));
}
