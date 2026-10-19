@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    var img: vec4f;
    if (p.filtering == 0) { img = texel_image(vec2i(floor(uv * vec2f(size_image())))); }
    else { img = sample_image(uv); }
    var o = img;
    if (p.composite == 1) {
        var bg = p.background_rgba;
        if (BACKGROUND_CONNECTED) { bg = load_background(c); }
        let k = img.a * p.tint.a;
        o = vec4f(mix(bg.rgb, img.rgb * p.tint.rgb, k), max(bg.a, k));
    }
    if (OUT_OUTPUT_GRAY) { let g = luminance(o.rgb); o = vec4f(g, g, g, 1.0); }
    store(c, o);
}
