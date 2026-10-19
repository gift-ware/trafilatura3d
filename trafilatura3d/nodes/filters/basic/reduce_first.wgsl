// First level of a reduction: each output texel summarises a factor.x * factor.y block of the input.
// kind 0: per-channel min (lo), max (hi), mean; kind 1/2: bounding box of the pixels whose colour
// (1) or alpha (2) is above 0, lo = (x0, y0), hi = (x1, y1) in source pixels (empty: lo > hi).
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
            let v = texel_input(q);
            if (p.kind == 0) {
                lo = min(lo, v);
                hi = max(hi, v);
                sum += v;
            } else {
                var on = max(v.r, max(v.g, v.b)) > 0.0;
                if (p.kind == 2) { on = v.a > 0.0; }
                if (on) {
                    lo = min(lo, vec4f(vec2f(q), 0.0, 0.0));
                    hi = max(hi, vec4f(vec2f(q), 0.0, 0.0));
                }
            }
        }
    }
    store_lo(c, lo);
    store_hi(c, hi);
    store_mean(c, sum / f32(p.factor.x * p.factor.y));
}
