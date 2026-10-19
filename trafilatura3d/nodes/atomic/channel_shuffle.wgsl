fn pick(a: vec4f, b: vec4f, i: i32) -> f32 {
    let v = select(b, a, i < 4);
    let k = i % 4;
    return select(select(select(v.a, v.b, k == 2), v.g, k == 1), v.r, k == 0);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var a = load_input_1(c);
    var b = load_input_2(c);
    if (INPUT_1_GRAY) { a = vec4f(a.r); }   // grayscale: alpha is the value too
    if (INPUT_2_GRAY) { b = vec4f(b.r); }
    store(c, vec4f(pick(a, b, p.red_channel), pick(a, b, p.green_channel), pick(a, b, p.blue_channel),
                   pick(a, b, p.alpha_channel)));
}
