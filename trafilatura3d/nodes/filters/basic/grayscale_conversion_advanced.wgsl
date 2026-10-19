@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c).rgb;
    let mx = max(v.r, max(v.g, v.b));
    let mn = min(v.r, min(v.g, v.b));
    var g = (mx + mn) * 0.5;
    switch p.grayscale_type {
        case 1: { g = luma709(v); }
        case 2: { g = (v.r + v.g + v.b) / 3.0; }
        case 3: { g = mx; }
        case 4: { g = mn; }
        default: {}
    }
    store(c, gray4(g));
}
