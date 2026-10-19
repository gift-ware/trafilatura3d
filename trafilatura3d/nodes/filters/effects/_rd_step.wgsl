// init: the contrasted input; else one activator-inhibitor step.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    if (p.init == 1) {
        let k = 1.0 + p.contrast * 8.0;
        store_out(c, fx_gray(clamp((load_input(c).r - 0.5) * k + 0.5, 0.0, 1.0)));
        return;
    }
    let gain = 6.0 + p.contrast * 18.0;
    let v = 0.5 + (texel_small(c).r - texel_large(c).r) * gain;
    store_out(c, fx_gray(clamp(v, 0.0, 1.0)));
}
