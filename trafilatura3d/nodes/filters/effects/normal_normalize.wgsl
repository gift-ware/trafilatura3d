@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = load_input(c);
    let v = t.xyz * 2.0 - 1.0;
    let l = length(v);
    let n = select(vec3f(0.0, 0.0, 1.0), v / l, l > 1e-6);
    store(c, vec4f(n * 0.5 + 0.5, t.a));
}
