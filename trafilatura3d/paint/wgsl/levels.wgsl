// The Levels effect (Levels page): input levels (shadows, midtones, highlights) then output levels (black and
// white points), on RGB or one component; Clamp keeps 0-1 (non HDR channels). Midtones: relative position
// between shadows and highlights, 0.5 = linear. Alpha (coverage) is kept.

fn level(x: f32) -> f32 {
    var t = (x - p.in_low) / max(p.in_high - p.in_low, 1e-6);
    if (p.clamp01 != 0) { t = clamp(t, 0.0, 1.0); }
    let g = log(0.5) / log(clamp(p.in_mid, 1e-3, 1.0 - 1e-3));
    let curved = sign(t) * pow(abs(t), g);
    let v = mix(p.out_low, p.out_high, curved);
    return select(v, clamp(v, 0.0, 1.0), p.clamp01 != 0);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    let v = load_image(c);
    var o = v.rgb;
    switch p.component {
        case 1: { o.r = level(v.r); }
        case 2: { o.g = level(v.g); }
        case 3: { o.b = level(v.b); }
        default: { o = vec3f(level(v.r), level(v.g), level(v.b)); }
    }
    store(c, vec4f(o, v.a));
}
