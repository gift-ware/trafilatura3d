// One doubling pass: the best of this pixel and the pixel `step_` pixels back along the direction, minus the
// fall over that step.
fn val(t: vec4f) -> f32 { return select(luminance(t.rgb), t.r, STATE_GRAY); }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (any(c >= size())) { return; }
    let reach = max(p.distance * mix(1.0, load_distance_map(c).r, p.distance_map_multiplier) * min(p.size_.x, p.size_.y), 1e-3);
    let back = sample_state(uv_of(c) - dir_of(p.angle) * p.step_ / p.size_);
    let g = max(val(load_state(c)), val(back) - p.step_ / reach);
    store_out(c, vec4f(g, g, g, 1.0));
}
