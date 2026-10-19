fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    var h = (v.rgb + p.offset) * p.multiply;  // the documentation: offset (add), then multiply
    if (p.invert != 0) { h = -h; }
    return vec4f(h, v.a);
}
