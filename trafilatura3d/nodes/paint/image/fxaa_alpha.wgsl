// The input's alpha as a grayscale image (for its own FXAA pass).
fn effect(c: vec2i) -> vec4f { let a = load_input(c).a; return vec4f(a, a, a, 1.0); }
