@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c).r;
    let disp = mix(load_blurred(c).r, 0.498039, clamp(p.relief / 32.0, 0.0, 1.0));
    store_displacement(c, gray4(disp));
    store_relief(c, gray4(clamp(v - disp + 0.5, 0.0, 1.0)));
}
