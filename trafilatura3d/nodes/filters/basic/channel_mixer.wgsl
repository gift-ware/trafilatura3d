fn mix_channel(v: vec3f, w: vec4f) -> f32 {
    return clamp((dot(v, w.xyz) + w.w) / 100.0, 0.0, 1.0);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c);
    let r = mix_channel(v.rgb, p.red_channel);
    var o = vec3f(r, mix_channel(v.rgb, p.green_channel), mix_channel(v.rgb, p.blue_channel));
    if (p.monochrome != 0) { o = vec3f(r); }
    store(c, vec4f(o, v.a));
}
