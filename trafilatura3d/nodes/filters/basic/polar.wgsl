@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    var q: vec2f;
    if (TO_POLAR) {
        let v = uv - 0.5;
        q = vec2f(fract(atan2(v.y, v.x) / (2.0 * PI) + 1.0), length(v) * 2.0);
    } else {
        q = vec2f(0.5) + rot(vec2f(uv.y * 0.5, 0.0), uv.x);
    }
    if (TO_POLAR) {
        // Seam-free across the angle wrap: pick texels directly, wrapping x.
        let s = vec2f(size_input());
        let x = q * s - 0.5;
        let i = vec2i(floor(x));
        let f = fract(x);
        let w = size_input().x;
        let x0 = wrap1(i.x, w);
        let x1 = (x0 + 1) % w;
        let a = mix(texel_input(vec2i(x0, i.y)), texel_input(vec2i(x1, i.y)), f.x);
        let b = mix(texel_input(vec2i(x0, i.y + 1)), texel_input(vec2i(x1, i.y + 1)), f.x);
        store(c, mix(a, b, f.y));
    } else {
        store(c, sample_input(q));
    }
}
