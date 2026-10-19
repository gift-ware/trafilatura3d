@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let lo = texel_lo(vec2i(0));
    let hi = texel_hi(vec2i(0));
    let mn = min(lo.r, min(lo.g, lo.b));
    let mx = max(hi.r, max(hi.g, hi.b));
    let v = load_input(c);
    var o = v.rgb;
    if (mx - mn > 1e-6) { o = (v.rgb - mn) / (mx - mn); }
    store(c, vec4f(o, v.a));
}
