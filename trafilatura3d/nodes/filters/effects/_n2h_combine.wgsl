@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = fx_uv(c);
    let hi = texel_h(c).r;
    let mid = sample_mid(uv).r;
    let lo = sample_low(uv).r;
    let b = 2.0 * clamp(p.balance, vec3f(0.0), vec3f(1.0));
    let v = lo * b.x + (mid - lo) * b.y + (hi - mid) * b.z;
    store_out(c, fx_gray(0.5 + v * 25.6));
}
