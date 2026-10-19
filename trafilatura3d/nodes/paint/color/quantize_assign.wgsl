// Quantize, Lloyd step 1: each grid cell takes the index of its nearest valid center.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let x = texel_down(c).xyz;
    var best = 0;
    var dist = 1e30;
    for (var k = 0; k < p.k_; k++) {
        let e = texel_palette(vec2i(k, 0));
        let d = dot(x - e.xyz, x - e.xyz);
        if (e.w >= 0.0 && d < dist) { dist = d; best = k; }
    }
    store(c, vec4f(f32(best), 0.0, 0.0, 1.0));
}
