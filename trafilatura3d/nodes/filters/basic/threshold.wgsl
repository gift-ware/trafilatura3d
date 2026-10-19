@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c).r;
    let t = p.threshold;
    var on = v > t;
    switch p.mode {
        case 1: { on = v >= t; }
        case 2: { on = v < t; }
        case 3: { on = v <= t; }
        default: {}
    }
    store(c, gray4(select(0.0, 1.0, on)));
}
