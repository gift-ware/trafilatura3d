@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let d1 = load_splatter_data_1(c);
    let d2 = load_splatter_data_2(c);
    store_pattern_id(c, vec4f(vec3f(round(d1.b) / 8.0), 1.0));
    store_pattern_uv(c, vec4f(d1.rg, d2.a, 1.0));
    store_random(c, vec4f(vec3f(d1.a), 1.0));
    store_transform(c, vec4f(d2.rgb, 1.0));
}
