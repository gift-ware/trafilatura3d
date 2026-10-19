// Seeds of both sides of a mask: pixels of `src` at 0.5 or more are "white" seeds (xy), the others "black" (zw).
fn effect(c: vec2i) -> vec4f {
    let me = vec2f(c);
    if (load_src(c).r >= 0.5) { return vec4f(me, -1.0, -1.0); }
    return vec4f(-1.0, -1.0, me);
}
