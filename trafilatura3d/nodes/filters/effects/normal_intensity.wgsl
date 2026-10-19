@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = load_input(c);
    let xy = t.xy * 2.0 - 1.0;
    let s = length(xy);
    if (s < 0.01) { store(c, vec4f(0.5, 0.5, 1.0, t.a)); return; }
    // The normal's angle from straight up, scaled down towards 0 or pushed towards a right angle; same direction.
    let a = asin(min(s, 1.0));
    let i = p.intensity;
    let b = select(a + pow(max(i, 0.0), 1.2 + i) * (1.5707964 - a), a * (1.0 + i), i <= 0.0);
    let n = vec3f(xy / s * sin(b), cos(b));
    store(c, vec4f(n * 0.5 + 0.5, t.a));
}
