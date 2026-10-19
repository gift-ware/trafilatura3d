// Box reduction: each output texel averages its p.factor.x x p.factor.y block of the input (a block cut by the
// border averages the texels it has).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let s = size_input();
    var acc = vec4f(0.0);
    var n = 0.0;
    for (var j = 0; j < p.factor.y; j++) {
        for (var i = 0; i < p.factor.x; i++) {
            let q = c * p.factor + vec2i(i, j);
            if (q.x < s.x && q.y < s.y) { acc += texel_input(q); n += 1.0; }
        }
    }
    store(c, acc / max(n, 1.0));
}
