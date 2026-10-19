// One doubling pass of the segmented OR-scans along rows (vertical_ 0) or columns, both ways at once: a pixel is
// reached from the left if it is, or if the window step_ further left is and no outline lies in between.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    let d = size();
    if (any(c >= d)) { return; }
    let me = texel_scan(c);
    var o = me;
    let off = select(vec2i(p.step_, 0), vec2i(0, p.step_), p.vertical_ == 1);
    let l = c - off;
    if (all(l >= vec2i(0))) {
        let L = texel_scan(l);
        o.x = max(me.x, (1.0 - me.y) * L.x);
        o.y = max(me.y, L.y);
    }
    let r = c + off;
    if (all(r < d)) {
        let R = texel_scan(r);
        o.z = max(me.z, (1.0 - me.w) * R.z);
        o.w = max(me.w, R.w);
    }
    store_scan(c, o);
}
