@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let fg = load_normal_fg(c);
    let bg = load_normal_bg(c);
    var k = clamp(p.opacity, 0.0, 1.0);
    if (p.use_mask == 1) { k *= clamp(load_mask(c).r, 0.0, 1.0); }
    let v = mix(bg.xyz * 2.0 - 1.0, fg.xyz * 2.0 - 1.0, k);
    let l = length(v);
    let n = select(vec3f(0.0, 0.0, 1.0), v / l, l > 1e-6);
    store(c, vec4f(n * 0.5 + 0.5, mix(bg.a, fg.a, k)));
}
