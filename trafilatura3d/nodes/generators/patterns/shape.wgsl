@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var uv = frame(c, p.non_square_expansion);
    let fs = frame_size(p.non_square_expansion);
    var inside = true;
    if (p.non_square_expansion == 1 && p.non_square_tiling == 0) {
        // One row/column of shapes centred on the longer axis.
        uv = uv - (fs - 1.0) * 0.5;
        inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
    }
    let n = f32(max(p.tiling, 1));
    var t = fract(uv * n);
    if (p.rotation_45 == 1) {
        // Before the tiling: the image turned 45 degrees and shrunk by 1/sqrt(2), repeated (the
        // shape in the middle, a quarter of one in each corner)
        t = fract(vec2f(t.x - t.y, t.x + t.y - 1.0) + 0.5);
    }
    let q0 = t * 2.0 - 1.0;
    let k = max(vec2f(p.scale) * p.size, vec2f(1e-4));
    let q = (rot(-p.angle) * q0) / k;
    var v = 0.0;
    if (inside && all(abs(q) <= vec2f(1.0))) { v = pattern_shape(p.pattern, q, p.pattern_specific); }
    store(c, vec4f(v, v, v, 1.0));
}
