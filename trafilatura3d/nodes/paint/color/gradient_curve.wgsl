fn bell(x: f32, center: f32, width: f32) -> f32 {
    let d = clamp(abs(x - center) / width, 0.0, 1.0);
    return 1.0 - d * d * (3.0 - 2.0 * d);
}

fn preset_curve(x: f32) -> f32 {
    switch p.preset {
        case 0: { return x; }
        case 1: { return x * x * (3.0 - 2.0 * x); }
        case 2: { return pow(x, 2.5); }
        case 3: { return 1.0 - pow(1.0 - x, 2.5); }
        case 4: { return bell(x, 0.5, 0.5); }
        case 5: { return 1.0 - bell(x, 0.5, 0.5); }
        case 6: { return max(bell(x, 0.25, 0.25), bell(x, 0.75, 0.25)); }
        default: {  // 4 steps with smooth risers
            let s = x * 4.0;
            let f = clamp((fract(s) - 0.8) / 0.2, 0.0, 1.0);
            return min((floor(s) + f * f * (3.0 - 2.0 * f)) / 4.0, 1.0);
        }
    }
}

fn effect(c: vec2i) -> vec4f {
    let v = load_input(c);
    let x = clamp(select(luminance(v.rgb), v.r, INPUT_GRAY), 0.0, 1.0);
    let y = balance_contrast(preset_curve(x), p.balance, p.contrast, p.invert);
    return vec4f(vec3f(y), 1.0);
}
