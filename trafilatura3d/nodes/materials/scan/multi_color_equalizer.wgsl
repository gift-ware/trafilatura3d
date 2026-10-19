@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let v = load_$in(c);
    var o = equalize(v.rgb, sample_blurred(uv).rgb, load_avg(c).rgb, load_m(c).r);
    if ($idx > p.input_count) { o = vec3f(0.0); }
    store_$out(c, vec4f(o, v.a));
}
