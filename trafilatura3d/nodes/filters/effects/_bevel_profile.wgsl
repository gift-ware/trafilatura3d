// Bevel height: a linear ramp over 128 x |Distance| pixels at 256 from the shapes'
// edge, by Euclidean distance (Round corners) or by the larger axis offset to the same nearest pixel
// (Angular corners). Distance > 0: 1 on the shapes, falling to 0 outside; < 0: 0 off them, rising to
// 1 inside. Use Custom Curve maps the ramp through the curve input.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let s = texel_seeds(c);
    let reach = max(0.5 * abs(p.distance) * p.size_.x, 1e-3);
    let outward = p.distance >= 0.0;
    // Nearest pixel on the other side: a white one for the outward ramp, a black one for the inward.
    let near = select(s.zw, s.xy, outward);
    var x = 0.0;
    let inside = all(s.xy == vec2f(c));
    if (near.x < 0.0) { x = 1e6; }
    else if (inside != outward) {
        let o = abs(fx_wrap_delta(vec2f(c), near));
        x = select(length(o), max(o.x, o.y), p.corner_type == 1) - 0.5;
    }
    var t = clamp(x / reach, 0.0, 1.0);
    if (outward) { t = 1.0 - t; }
    if (p.use_custom_curve == 1) { t = sample_custom_curve(vec2f(t, 0.5)).r; }
    store_out(c, fx_gray(t));
}
