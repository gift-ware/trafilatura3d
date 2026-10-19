// Quantize, Lloyd step 2: each valid center moves to the mean of its cells; count, per-axis variance and sum of
// squared distances are recorded. An empty cluster keeps its center with count 0.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let old = texel_palette(vec2i(c.x, 0));
    var center = old;
    var stats = vec4f(0.0);
    if (old.w >= 0.0) {
        let n = size_down();
        var sum = vec3f(0.0);
        var sq = vec3f(0.0);
        var count = 0.0;
        for (var y = 0; y < n.y; y++) {
            for (var x = 0; x < n.x; x++) {
                if (i32(texel_index(vec2i(x, y)).r + 0.5) == c.x) {
                    let v = texel_down(vec2i(x, y)).xyz;
                    sum += v; sq += v * v; count += 1.0;
                }
            }
        }
        if (count > 0.0) {
            let m = sum / count;
            let var3 = max(sq / count - m * m, vec3f(0.0));
            center = vec4f(m, count);
            stats = vec4f(var3, (var3.x + var3.y + var3.z) * count);
        } else {
            center = vec4f(old.xyz, 0.0);
        }
    }
    store_palette(vec2i(c.x, 0), center);
    store_palette(vec2i(c.x, 1), stats);
}
