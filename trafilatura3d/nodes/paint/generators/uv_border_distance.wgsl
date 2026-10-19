fn effect(c: vec2i) -> vec4f {
    let s = load_seeds_(c);
    let inside = load_uv_islands(c).a >= 0.5;
    var v = 0.0;
    if (!inside) {
        v = 1.0;
    } else if (s.x >= 0.0) {
        let d = distance(vec2f(c), s.xy) * 256.0 / p.size_.x;  // px at 256
        let t = clamp(d / max(p.distance, 1e-4), 0.0, 1.0);
        let round_ = sqrt(max(1.0 - t * t, 0.0));
        let smooth_ = 1.0 - smoothstep(0.0, 1.0, t);
        v = select(mix(round_, smooth_, p.smoothness), 0.0, t >= 1.0);
    }
    v = g_contrast(g_balance(v, p.balance), p.contrast);
    return vec4f(vec3f(g_invert(v, p.invert)), 1.0);
}
