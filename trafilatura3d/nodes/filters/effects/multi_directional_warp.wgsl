@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    let amount = load_intensity_input(c).r * p.intensity / 256.0;
    if (p.pass_angle >= 0.0) {
        store(c, sample_input(uv + fx_dir(p.pass_angle) * amount));
        return;
    }
    let n = select(select(8, 4, p.directions == 1), 2, p.directions == 0);
    let mode = select(p.mode, 0, n == 2);
    var acc = vec4f(0.0);
    var mx = vec4f(-1e30);
    var mn = vec4f(1e30);
    for (var k = 0; k < n; k++) {
        let s = sample_input(uv + fx_dir(p.warp_angle + f32(k) / f32(n)) * amount);
        acc += s;
        mx = max(mx, s);
        mn = min(mn, s);
    }
    var o = acc / f32(n);
    if (mode == 1) { o = mx; }
    if (mode == 2) { o = mn; }
    store(c, o);
}
