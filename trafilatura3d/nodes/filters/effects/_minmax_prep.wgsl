// Value and its negation, to reduce to (max, -min) with a max reduction.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let t = load_src(c);
    let v = select(luminance(t.rgb), t.r, SRC_GRAY);
    store_out(c, vec4f(v, -v, 0.0, 0.0));
}
