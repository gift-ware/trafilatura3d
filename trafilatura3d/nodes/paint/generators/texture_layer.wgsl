// A custom texture layer (template: $port is the input, $PORT its upper case): repeated over the UVs Scale
// times (square tiles unless Non-Square Tiling, then stretched to the Texture Set's proportions) or projected
// from the three axes on the position map (Triplanar), blended by the world space normals raised to the
// Triplanar Contrast; converted to grayscale, inverted, contrast and brightness. Returns (value, 1), or (0, 0)
// when nothing is connected (the layer is then left out). Needs the inputs `position` and `world_space_normals`.

fn wrapped_$port(uv: vec2f) -> vec4f {
    let d = size_$port();
    let x = uv * vec2f(d) - 0.5;
    let i = vec2i(floor(x));
    let f = fract(x);
    return mix(mix(texel_$port(wrap(i, d)), texel_$port(wrap(i + vec2i(1, 0), d)), f.x),
               mix(texel_$port(wrap(i + vec2i(0, 1), d)), texel_$port(wrap(i + vec2i(1, 1), d)), f.x), f.y);
}

fn tex_$port(c: vec2i, scale: f32, triplanar: i32, tcontrast: f32, non_square: i32, method: i32, invert: i32,
             contrast: f32, brightness: f32) -> vec2f {
    if (!$PORT_CONNECTED) { return vec2f(0.0); }
    var rgb: vec3f;
    if (triplanar != 0) {
        let pos = load_position(c).rgb * scale;
        let n = abs(g_normal(load_world_space_normals(c)));
        var w = pow(n, vec3f(1.0 + tcontrast * 16.0));
        w /= max(w.x + w.y + w.z, 1e-6);
        rgb = w.x * wrapped_$port(pos.zy).rgb + w.y * wrapped_$port(pos.xz).rgb + w.z * wrapped_$port(pos.xy).rgb;
    } else {
        var uv = uv_of(c) * scale;
        if (non_square == 0) { uv.x *= p.size_.x / p.size_.y; }
        rgb = wrapped_$port(uv).rgb;
    }
    var v = select(g_gray(rgb, method), rgb.r, $PORT_GRAY);
    v = g_invert(v, invert);
    v = clamp(g_contrast(v, contrast) + brightness, 0.0, 1.0);
    return vec2f(v, 1.0);
}
