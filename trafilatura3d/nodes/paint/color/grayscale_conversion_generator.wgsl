fn effect(c: vec2i) -> vec4f {
    var deg = p.rotation;
    if (p.safe_rotation != 0) { deg = round(deg / 45.0) * 45.0; }
    let a = radians(deg);
    let d = uv_of(c) - 0.5;
    let r = vec2f(cos(a) * d.x + sin(a) * d.y, -sin(a) * d.x + cos(a) * d.y);
    let uv = fract((r + 0.5) * f32(max(p.tile, 1)));
    let v = clamp(sample_source_color(uv).rgb, vec3f(0.0), vec3f(1.0));
    return vec4f(vec3f(balance_contrast(gray_of(v, p.grayscale_type), p.balance, p.contrast, p.invert)), 1.0);
}
