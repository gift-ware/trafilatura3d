// One direction of a gaussian blur (radius ~ 2 sigma, output pixels). Large radii take at most 48
// bilinear taps per side, spaced evenly. channel: 0 as is, 1 alpha, 2 luminance.
fn pick(t: vec4f) -> vec4f {
    if (p.channel == 1) { return vec4f(t.a, t.a, t.a, 1.0); }
    if (p.channel == 2) { let l = select(luminance(t.rgb), t.r, SRC_GRAY); return vec4f(l, l, l, 1.0); }
    return t;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c) + p.shift;
    let r = max(p.radius, 0.0);
    if (r < 0.5) { store_out(c, pick(sample_src(uv))); return; }
    let dir = select(vec2f(1.0, 0.0), vec2f(0.0, 1.0), p.vertical == 1) / p.size_;
    let n = i32(min(ceil(r), 48.0));
    let stride = r / f32(n);
    let sigma = r * 0.5;
    var acc = vec4f(0.0);
    var wsum = 0.0;
    for (var k = -n; k <= n; k++) {
        let x = f32(k) * stride;
        let w = exp(-0.5 * x * x / (sigma * sigma));
        acc += w * pick(sample_src(uv + dir * x));
        wsum += w;
    }
    store_out(c, acc / wsum);
}
