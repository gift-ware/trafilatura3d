@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var uv = frame(c, p.non_square_expansion);
    let fs = frame_size(p.non_square_expansion);
    var inside = true;
    if (p.non_square_expansion == 1 && p.non_square_tiling == 0) {
        uv = uv - (fs - 1.0) * 0.5;
        inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
    }
    let nt = f32(max(p.tiling, 1));
    let q = rot(-p.rotation) * (fract(uv * nt) * 2.0 - 1.0);
    let n = f32(max(p.sides, 3));
    let h = PI / n;
    var radius = p.scale;                                  // circumradius
    if (p.auto_scale == 0) { radius = p.scale / cos(h); }  // apothem = scale
    let ap = radius * cos(h);
    let ts = max(p.triangle_size, vec2f(1e-3));
    let e = p.explode * radius * 0.5;
    let ac = ap * 2.0 / 3.0;
    let k0 = i32(round(atan2(q.y, q.x) / (2.0 * h)));     // slice 0 points right (a flat edge there)
    var v = 0.0;
    for (var k = k0 - 1; k <= k0 + 1; k++) {
        let phi = f32(k) * 2.0 * h;
        let dir = vec2f(cos(phi), sin(phi));
        let a = dot(q, dir);
        let b = dot(q, vec2f(-dir.y, dir.x));
        let a1 = (a - e - ac) / ts.x + ac;
        let b1 = b / ts.y;
        let sd = min((ap - a1) * ts.x, (a1 * sin(h) - abs(b1) * cos(h)) * min(ts.x, ts.y));
        var value = 1.0;
        if (p.gradient == 1) {
            let g = clamp(a1 / max(ap, 1e-5), 0.0, 1.0);
            value = select(1.0 - g, g, p.gradient_invert == 1);
        }
        v = max(v, select(0.0, value, sd >= 0.0));          // hard edges
    }
    if (!inside) { v = 0.0; }
    store(c, vec4f(v, v, v, 1.0));
}
