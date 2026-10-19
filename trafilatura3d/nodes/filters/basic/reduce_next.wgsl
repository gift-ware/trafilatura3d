// Further levels of a reduction (see reduce_first.wgsl).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var lo = vec4f(1e30);
    var hi = vec4f(-1e30);
    var sum = vec4f(0.0);
    for (var j = 0; j < p.factor.y; j++) {
        for (var i = 0; i < p.factor.x; i++) {
            let q = c * p.factor + vec2i(i, j);
            lo = min(lo, texel_lo(q));
            hi = max(hi, texel_hi(q));
            sum += texel_mean(q);
        }
    }
    store_lo(c, lo);
    store_hi(c, hi);
    store_mean(c, sum / f32(p.factor.x * p.factor.y));
}
