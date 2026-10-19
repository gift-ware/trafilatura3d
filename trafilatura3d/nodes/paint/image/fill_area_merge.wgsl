// After the scans of one axis: exterior = reached from either side; restart the scan state for the other axis.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let s = texel_scan(c);
    let b = texel_outline(c).r;
    let e = select(0.0, 1.0, b < 0.5 && max(s.x, s.z) > 0.5);
    store_scan(c, vec4f(e, b, e, b));
}
