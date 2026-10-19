fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }

// u from v on the better conditioned axis.
fn solve_u(h: vec2f, e: vec2f, f: vec2f, g: vec2f, v: f32) -> f32 {
    let den = e + g * v;
    let num = h - f * v;
    return select(num.y / den.y, num.x / den.x, abs(den.x) > abs(den.y));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let pt = uv_at(c);
    let a = p.p00;
    let e = p.p10 - a;
    let f = p.p01 - a;
    let g = a - p.p10 + p.p11 - p.p01;
    let h = pt - a;
    let k2 = cross2(g, f);
    let k1 = cross2(e, f) + cross2(h, g);
    let k0 = cross2(h, e);
    var roots: array<vec2f, 2>;
    var count = 0;
    if (abs(k2) < 1e-7) {
        if (abs(k1) > 1e-9) {
            let v = -k0 / k1;
            roots[0] = vec2f(solve_u(h, e, f, g, v), v);
            count = 1;
        }
    } else {
        let w = k1 * k1 - 4.0 * k0 * k2;
        if (w >= 0.0) {
            let sw = sqrt(w);
            let v1 = (-k1 - sw) / (2.0 * k2);
            let v2 = (-k1 + sw) / (2.0 * k2);
            roots[0] = vec2f(solve_u(h, e, f, g, v1), v1);
            roots[1] = vec2f(solve_u(h, e, f, g, v2), v2);
            count = 2;
        }
    }
    var front = vec2f(0.0);
    var back = vec2f(0.0);
    var has_front = false;
    var has_back = false;
    for (var i = 0; i < count; i++) {
        let r = roots[i];
        let inside = all(r >= vec2f(0.0)) && all(r <= vec2f(1.0));
        if (!inside && p.enable_tiling == 0) { continue; }
        let det = cross2(e + g * r.y, f + g * r.x);
        if (det >= 0.0) { if (!has_front) { front = r; has_front = true; } }
        else if (!has_back) { back = r; has_back = true; }
    }
    var show = false;
    var r = vec2f(0.0);
    switch p.culling {
        case 0: { show = has_front; r = front; }
        case 1: { show = has_back; r = back; }
        case 2: { show = has_front || has_back; r = select(back, front, has_front); }
        default: { show = has_front || has_back; r = select(front, back, has_back); }
    }
    if (!show) {
        store(c, background(p.background_color, INPUT_GRAY));
        return;
    }
    let q = select(r, fract(r), p.enable_tiling != 0);
    if (p.sampling == 1) { store(c, texel_input(vec2i(floor(q * vec2f(size_input()))))); }
    else { store(c, sample_input(q)); }
}
