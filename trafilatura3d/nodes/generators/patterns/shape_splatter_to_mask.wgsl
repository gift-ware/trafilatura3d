@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let d1 = load_splatter_data_1(c);
    let d2 = load_splatter_data_2(c);
    let pid = i32(round(d1.b));
    let lo = min(p.pattern_id_start_range, p.pattern_id_end_range);
    let hi = max(p.pattern_id_start_range, p.pattern_id_end_range);
    // Random masking per tile (key: tile index and random).
    let key = vec2i(i32(round(d2.a * 1048576.0)), i32(round(d1.a * 1048576.0)));
    let sel = pid >= 1 && pid >= lo && pid <= hi && rnd(key, 7u) >= p.random_mask;
    var v = 0.0;
    if (sel) {
        switch p.output_mode {
            case 1: { v = f32(pid); }
            case 2: { v = f32(pid - lo + 1) / f32(hi - lo + 1); }
            default: { v = 1.0; }
        }
    }
    store(c, vec4f(v, v, v, 1.0));
}
