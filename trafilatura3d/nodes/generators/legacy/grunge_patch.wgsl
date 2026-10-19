// The Brush pattern of the grunges (ours): the image kept inside a centred rectangle, 0.83 of the image wide and
// shorter as Brush pattern grows (`half_height`), with a soft border `soft` wide that the image itself frays:
// brighter parts reach `fray` further out.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_input(c).r;
    let q = abs((vec2f(c) + 0.5) / vec2f(d) - 0.5);
    let inside = min(0.415 - q.x, p.half_height - q.y) + p.fray * (v - 0.5);
    let m = clamp(inside / p.soft, 0.0, 1.0);
    store(c, vec4f(v * m, v * m, v * m, 1.0));
}
