@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = clamp(p.inputs, 2, 16);
    var acc = vec3f(0.0);
    var wsum = 0.0;
    var darkest = vec3f(0.0);   // fallbacks when every exposure is clipped or crushed
    var brightest = vec3f(0.0);
    var v: vec3f;
    var i: i32;
//@each
    i = $i;
    v = load_$ch(c).rgb;
    if (i < n) {
        let lin = srgb2lin(v);
        let scale = exp2(f32(n - 1 - i) * p.exposure_delta);
        let x = sat(max(v.r, max(v.g, v.b)));
        let w = max(1.0 - pow(abs(2.0 * x - 1.0), 12.0), 0.0);
        acc += w * lin * scale;
        wsum += w;
        if (i == 0) { darkest = lin * scale; }
        if (i == n - 1) { brightest = lin * scale; }
    }
//@end
    var rad = acc / max(wsum, 1e-6);
    if (wsum < 1e-4) { rad = select(brightest, darkest, luma(brightest) > 0.5); }
    rad = min(rad, vec3f(exp2(p.white_point)));
    store(c, vec4f(rad, 1.0));
}
