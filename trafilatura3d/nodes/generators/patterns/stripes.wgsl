@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let n = f32(max(p.stripes, 1));
    let sh = f32(p.shift);
    let y = select(uv.y, uv.y - 0.5, p.align == 1);
    let phase = uv.x * n - y * sh;                    // Shift > 0 leans them like a backslash
    let g = length(vec2f(n, sh));                       // phase units per uv
    let f = fract(phase);
    let hw = 0.5 * clamp(p.width, 0.0, 1.0);
    let dist = hw - abs(f - 0.5);                       // phase units, > 0 inside
    let s = p.softness * hw;
    var v: f32;
    if (p.filtering == 1) { v = ramp(dist, s, g * pixel_uv()); }
    else { v = select(clamp(dist / max(s, 1e-6), 0.0, 1.0), step(0.0, dist), s <= 0.0); }
    if (p.width >= 1.0) { v = 1.0; }
    store(c, vec4f(v, v, v, 1.0));
}
