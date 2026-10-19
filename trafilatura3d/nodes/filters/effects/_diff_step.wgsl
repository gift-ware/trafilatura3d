@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    if (diff_known(c)) {
        store_v(c, diff_encode(load_source(c)));
        store_w(c, fx_gray(1.0));
        return;
    }
    let reach = p.reach * clamp(load_intensity(c).r, 0.0, 1.0);
    var ang = 0.0;
    if (p.dither == 1) { ang = hash21(c, f32(p.pass_) + 0.5) * 2.0 * PI; }
    let w0 = texel_w(c).r;
    var acc = texel_v(c) * w0;
    var wsum = w0;
    for (var k = 0; k < 8; k++) {
        var o = fx_rot(vec2f(1.0, 0.0), ang + f32(k) * PI * 0.25);
        if (k % 2 == 1 && p.dither == 0) { o *= 1.41421356; }
        let q = c + vec2i(round(o * reach));
        let wq = texel_w(q).r;
        acc += texel_v(q) * wq;
        wsum += wq;
    }
    var v = select(vec4f(0.0), acc / wsum, wsum > 1e-12);
    if (p.normal == 1 && p.mode == 0 && wsum > 1e-12) {
        let n = v.xyz * 2.0 - 1.0;
        let l = length(n);
        if (l > 1e-6) { v = vec4f(n / l * 0.5 + 0.5, v.a); }
    }
    store_v(c, v);
    store_w(c, fx_gray(wsum / 9.0));
}
