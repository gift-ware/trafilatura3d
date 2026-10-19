@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let fs = frame_size(p.non_square_expansion);
    let uv = frame(c, p.non_square_expansion);
    // Tiling: each repetition is the whole image, centred in its cell (a half-cell offset on even
    // counts, as Shape).
    let tn = f32(max(p.tiling, 1));
    let t = (uv - 0.5 * fs) * tn + select(0.0, 0.5, p.tiling % 2 == 0);
    let q = rot(-p.rotation) * (select(t, fract(t + 0.5) - 0.5, tn > 1.0) * 2.0);
    let n = f32(max(p.sides, 3));
    let h = PI / n;
    var radius = p.scale;
    if (p.auto_scale == 0) { radius = p.scale / cos(h); }
    let ap = max(radius * cos(h), 1e-5);
    // Fold into the sector whose bisector is nearest, then distance to its edge.
    let k = round(atan2(q.y, q.x) / (2.0 * h));          // sector 0 points right (a flat edge there)
    let phi = k * 2.0 * h;
    let a = dot(q, vec2f(cos(phi), sin(phi)));
    let sd = ap - a;
    let cover = edge(sd, 2.0 * tn * pixel_uv());
    var v = clamp(sd / ap / max(p.gradient, 1e-3), 0.0, 1.0);
    v = pow(v, exp2(-2.0 * p.curve));
    if (p.invert_gradient == 1) { v = 1.0 - v; }
    v *= cover;
    store(c, vec4f(v, v, v, 1.0));
}
