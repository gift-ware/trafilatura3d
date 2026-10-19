@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e = 1.0 / vec2f(d);
    let hc = sample_position(uv).r;
    let up = up_dir(hc, sample_position(uv + vec2f(e.x * 2.0, 0.0)).r, sample_position(uv + vec2f(0.0, e.y * 2.0)).r);
    let side = vec2f(-up.y, up.x);
    let len = p.length * 0.04;
    var v = sample_blurred(uv).r;
    for (var k = 1; k <= 32; k++) {
        let t = f32(k) / 32.0;
        v = max(v, sample_blurred(uv + up * t * len).r * (1.0 - t * 0.8));
    }
    let streak = vnoise(vec2f(dot(uv, side), dot(uv, up) * 0.03), vec2i(48, 4), p.seed_ + 5.0);
    v *= smoothstep(0.25, 0.75, streak) * 1.3;
    if (p.occlude < 3) {
        let n = wsn(load_normal_world_space(c).rgb);
        v *= mix(1.0, load_ambient_occlusion(c).r, abs(n[p.occlude]));
    }
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
