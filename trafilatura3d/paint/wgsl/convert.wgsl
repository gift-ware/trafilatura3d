// A channel between the Paint and Graph material conventions: RGB * scale + offset, green flipped.

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy) + origin();
    if (any(c >= size())) { return; }
    let v = load_image(c);
    var rgb = v.rgb * p.scale + p.offset;
    if (p.flip_green != 0) { rgb.g = 1.0 - rgb.g; }
    store(c, vec4f(rgb, v.a));
}
