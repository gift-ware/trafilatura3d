// Value of a strip at abscissa xs (in knots): its knots faded by Smoothness, linearly and with a
// Gaussian window (sigma half a knot at full Smoothness), mixed by Smoothness interpolation.
fn strip(xs: f32, row: i32, per: vec2i) -> f32 {
    let sm = max(p.smoothness, 1e-4);
    let i = i32(floor(xs));
    let f = fract(xs);
    let lin = mix(hw(vec2i(i, row), per, p.seed_), hw(vec2i(i + 1, row), per, p.seed_),
                  clamp((f - 0.5) / sm + 0.5, 0.0, 1.0));
    let sigma = 0.5 * sm;
    var gs = 0.0;
    var gw = 0.0;
    for (var k = -2; k <= 3; k++) {
        let w = exp(-0.5 * pow((f - f32(k)) / sigma, 2.0));
        gs += w * hw(vec2i(i + k, row), per, p.seed_);
        gw += w;
    }
    return mix(lin, gs / gw, clamp(p.smoothness_interpolation, 0.0, 1.0));
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    var uv = uv_of(c) * asp;
    var ax = asp;
    var ny = f32(max(p.y_amount, 1));
    if (p.y_amount_by_resolution != 0) { ny = p.size_.y; }
    if (p.rotate != 0) { uv = uv.yx; ax = asp.yx; if (p.y_amount_by_resolution != 0) { ny = p.size_.x; } }
    let per = vec2i(round(vec2f(f32(max(p.x_amount, 1)), ny) * ax));
    // Along the strip (x) values fade over Smoothness; across (y) strips are hard-edged. Each strip
    // starts at its own random phase, which Disorder slides further.
    let row = i32(floor(uv.y * f32(per.y) / ax.y));
    let phase = hw(vec2i(row, 0), vec2i(per.y, 1), p.seed_ + 3.0) * (1.0 + p.disorder * p.disorder_speed * 4.0);
    let xs = uv.x * f32(per.x) / ax.x + phase - 0.5;
    // The faded values span less than the knots: stretched back to the full range around mid-gray.
    let v = clamp((strip(xs, row, per) - 0.5) * 1.25 + 0.5, 0.0, 1.0);
    store(c, vec4f(v, v, v, 1.0));
}
