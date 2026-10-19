@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var t = load_grayscale_input(c).r;
    t = select(clamp(t, 0.0, 1.0), fract(t), p.gradient_addressing == 0);
    let n = vec2f(size_gradient_input());
    let along = clamp(t * (n - 1.0) + 0.5, vec2f(0.5), n - 0.5) / n;  // texel centres, no wrap
    let pos = p.gradient_input_position;
    let uv = select(vec2f(pos, along.y), vec2f(along.x, pos), p.gradient_orientation == 0);
    store(c, sample_gradient_input(uv));
}
