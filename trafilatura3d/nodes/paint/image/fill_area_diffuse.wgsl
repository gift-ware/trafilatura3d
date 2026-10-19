// Diffusion of the stroke colors into the closed areas: step_ 0 initializes (strokes = their color with weight 1,
// the rest weight 0); each pass, a pixel inside an area takes the weighted mean of itself and its 8 neighbours
// step_ pixels away (strokes and exterior never change).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    let d = size();
    if (any(c >= d)) { return; }
    let stroke = texel_outline(c).r > 0.5;
    if (p.step_ == 0) {
        store_colors(c, select(vec4f(0.0), vec4f(load_input(c).rgb, 1.0), stroke));
        return;
    }
    let me = texel_colors(c);
    if (stroke || texel_scan(c).r > 0.5) { store_colors(c, me); return; }
    var acc = vec3f(0.0);
    var w = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let q = c + vec2i(i, j) * p.step_;
            if (any(q < vec2i(0)) || any(q >= d)) { continue; }
            let s = texel_colors(q);
            acc += s.rgb * s.a;
            w += s.a;
        }
    }
    if (w <= 0.0) { store_colors(c, me); return; }
    store_colors(c, vec4f(acc / w, max(w / 9.0, 1e-3)));
}
