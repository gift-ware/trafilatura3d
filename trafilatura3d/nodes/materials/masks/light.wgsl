@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let n = wsn(load_world_space_normal(c).rgb);
    let h = p.horizontal_angle * TAU;
    let v = p.vertical_angle * 0.5 * PI;
    let l = vec3f(cos(v) * sin(h), sin(v), cos(v) * cos(h));
    let ndl = sat(dot(n, l));
    let spec = pow(ndl, 1.0 / max(1.0 - p.highlight_glossiness, 1e-3)) * p.highlight_level;
    store(c, g4(sat(ndl * (1.0 - 0.5 * p.highlight_level) + spec)));
}
