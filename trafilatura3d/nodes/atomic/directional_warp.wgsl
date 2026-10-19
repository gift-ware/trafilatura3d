@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = (vec2f(c) + 0.5) / vec2f(d);
    let a = p.warp_angle * 2.0 * PI;
    let amount = (load_intensity_input(c).r - p.intensity_map_offset) * p.intensity / 256.0;
    let q = uv + vec2f(cos(a), sin(a)) * p.pixel_size_ * amount;
    if (p.input_filtering_mode == 0) {
        store(c, texel_input(vec2i(floor(q * vec2f(size_input())))));
    } else {
        store(c, sample_input(q));
    }
}
