@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var t = load_input(c).r;
    if (p.gradient_adressing == 0) { t = fract(t); }
    let k = lut_gradient(t);
    if (OUT_OUTPUT_GRAY) { let g = luminance(k.rgb); store(c, vec4f(g, g, g, 1.0)); } else { store(c, k); }
}
