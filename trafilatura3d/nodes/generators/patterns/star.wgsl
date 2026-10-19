// Signed distance to a five-pointed star of outer radius r, inner ratio rf, point up (y up).
fn sd_star5(q: vec2f, r: f32, rf: f32) -> f32 {
    let k1 = vec2f(0.809016994375, -0.587785252292);
    let k2 = vec2f(-k1.x, k1.y);
    var p2 = vec2f(abs(q.x), q.y);
    p2 -= 2.0 * max(dot(k1, p2), 0.0) * k1;
    p2 -= 2.0 * max(dot(k2, p2), 0.0) * k2;
    p2.x = abs(p2.x);
    p2.y -= r;
    let ba = rf * vec2f(-k1.y, k1.x) - vec2f(0.0, 1.0);
    let h = clamp(dot(p2, ba) / dot(ba, ba), 0.0, r);
    return length(p2 - ba * h) * sign(p2.y * ba.x - p2.x * ba.y);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let fs = frame_size(p.non_square_expansion);
    let uv = frame(c, p.non_square_expansion);
    let r = max(p.scale, 1e-5);
    let q0 = (uv - 0.5 * fs) * 2.0;
    let q = vec2f(q0.x, -q0.y);                                  // y up, outer circle centred
    let v = edge(-sd_star5(q, r, 0.381966), pixel_uv());
    store(c, vec4f(v, v, v, 1.0));
}
