fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    var b = vec3f(0.0);
    for (var j = -1; j <= 1; j++) { for (var i = -1; i <= 1; i++) { b += texel_input(c + vec2i(i, j)).rgb; } }
    return vec4f(v.rgb + (v.rgb - b / 9.0) * p.sharpen_intensity * 2.0, v.a);
}
