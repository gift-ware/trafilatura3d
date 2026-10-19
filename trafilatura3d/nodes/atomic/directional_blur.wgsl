// One doubling pass of the directional box (directional_blur.py): the average of the input at uv + p.a_ and
// uv + p.b_ (offsets in uv).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    store(c, 0.5 * (sample_input(uv + p.a_) + sample_input(uv + p.b_)));
}
