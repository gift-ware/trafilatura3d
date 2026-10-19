// MatFinish mapping in texture space: the pattern over the UV square.
fn mf_sample(c: vec2i) -> vec4f { return pattern(uv_of(c)); }
fn mf_robust() -> bool { return false; }
