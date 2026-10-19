// Outline (value or alpha above thr_, or with edges_ a jump of at least thr_ to a neighbour) and the first exterior
// seeds: the open pixels of the texture's border. scan = (exterior from the left, blocked on the left, the same
// from the right) for the segmented scans.
fn val(q: vec2i) -> f32 {
    let t = texel_input(clamp(q, vec2i(0), size_input() - 1));
    if (p.alpha_ != 0) { return t.a; }
    return select(luminance(t.rgb), t.r, INPUT_GRAY);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    let d = size();
    if (any(c >= d)) { return; }
    let q = vec2i(floor((vec2f(c) + 0.5) * vec2f(size_input()) / p.size_));
    let v = val(q);
    var o = v >= p.thr_;
    if (p.edges_ != 0) {
        let jump = max(max(abs(v - val(q + vec2i(1, 0))), abs(v - val(q - vec2i(1, 0)))),
                       max(abs(v - val(q + vec2i(0, 1))), abs(v - val(q - vec2i(0, 1)))));
        o = jump >= p.thr_;
    }
    let border = c.x == 0 || c.y == 0 || c.x == d.x - 1 || c.y == d.y - 1;
    let e = select(0.0, 1.0, border && !o);
    let b = select(0.0, 1.0, o);
    store_scan(c, vec4f(e, b, e, b));
    store_outline(c, vec4f(b, b, b, 1.0));
}
