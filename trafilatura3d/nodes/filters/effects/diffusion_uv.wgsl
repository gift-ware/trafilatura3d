// Final pass: decode the diffused values (known pixels are already the source).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var o = diff_decode(texel_v(c));
    if (diff_known(c)) { o = select(load_source(c), diff_decode(diff_encode(load_source(c))), p.mode == 2); }
    if (p.alpha_mask == 1) { o.a = 1.0; }
    if (OUT_OUTPUT_GRAY) { o = fx_gray(o.r); }
    store(c, o);
}
