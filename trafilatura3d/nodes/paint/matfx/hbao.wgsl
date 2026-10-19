// The channel multiplied by the (blurred) occlusion, its coverage kept.
fn fx(c: vec2i) {
    let v = load_input(c);
    store(c, vec4f(v.rgb * sample_ao(mx_uv(c)).r, v.a));
}
