@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // A tent 1 - |v - Position| / Range, then Levels with input range Contrast / 2 ..
    // 1 - Contrast / 2 (Contrast 1: a hard edge at half the tent).
    let t = clamp(1.0 - abs(load_input(c).r - p.position) / max(p.range, 1e-5), 0.0, 1.0);
    let k = clamp(p.contrast, 0.0, 1.0) * 0.5;
    var o = select(0.0, 1.0, t >= 0.5);
    if (k < 0.5 - 1e-5) { o = clamp((t - k) / (1.0 - 2.0 * k), 0.0, 1.0); }
    store(c, gray4(o));
}
