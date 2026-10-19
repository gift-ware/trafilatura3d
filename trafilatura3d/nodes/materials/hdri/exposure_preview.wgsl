@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let n = clamp(i32(log2(f32(d.y))) - 4, 2, 8);
    let band = min(i32(uv.y * f32(n)), n - 1);
    let local = vec2f(uv.x, fract(uv.y * f32(n)));
    let ev = mix(p.max_exposure, p.min_exposure, f32(band) / f32(n - 1));
    let v = sample_input(local);
    store(c, vec4f(v.rgb * exp2(ev), 1.0));
}
