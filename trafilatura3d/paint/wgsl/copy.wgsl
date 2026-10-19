// Painted pixels into the stack: the entry as is, bilinearly resampled when its size differs.

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    var v: vec4f;
    if (all(size_image() == size())) {
        v = texel_image(c);
    } else {
        v = sample_image((vec2f(c) + 0.5) / p.size_);
    }
    if (p.gray != 0) { v = vec4f(vec3f(v.r), v.a); }
    store(c, v);
}
