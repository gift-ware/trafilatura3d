// 1D blur of `input` along p.direction (output pixels): box (half-width p.radius) or gaussian
// (sigma p.radius). Long kernels take at most p.taps bilinear taps per side, spread evenly.
// Along a row or a column of an input of the output's size a tap is a lerp of two texels on that line (the bilinear
// tap's other two have weight 0).
fn tap(c: vec2i, uv: vec2f, x: f32, along: vec2i) -> vec4f {
    if (any(along != vec2i(0))) {
        let q = c + i32(floor(x)) * along;
        return mix(texel_input(q), texel_input(q + along), fract(x));
    }
    return sample_input(uv + p.direction * x / vec2f(size()));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let r = p.radius;
    let extent = select(r + 0.5, 3.0 * r, p.kernel == 1);
    if (r < 0.05) { store(c, sample_input(uv)); return; }
    let n = clamp(i32(ceil(extent)), 1, max(p.taps, 1));
    let step = extent / f32(n);
    let lined = all(size_input() == d) && (p.direction.x == 0.0 || p.direction.y == 0.0);
    let along = select(vec2i(0), vec2i(sign(p.direction)), lined);
    var acc = vec4f(0.0);
    var wsum = 0.0;
    for (var k = -n; k <= n; k++) {
        let x = f32(k) * step;
        var w: f32;
        if (p.kernel == 1) { w = exp(-0.5 * x * x / (r * r)); }
        else { w = clamp((r + 0.5 - abs(x)) / step + 0.5, 0.0, 1.0); }
        acc += w * tap(c, uv, x, along);
        wsum += w;
    }
    store(c, acc / max(wsum, 1e-6));
}
