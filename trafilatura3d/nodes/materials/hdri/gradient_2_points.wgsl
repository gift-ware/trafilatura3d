@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dir = ll_dir(uv_of(c, d));
    let a1 = acos(clamp(dot(dir, ll_dir(p.point_1_position)), -1.0, 1.0));
    let a2 = acos(clamp(dot(dir, ll_dir(p.point_2_position)), -1.0, 1.0));
    let t = a1 / max(a1 + a2, 1e-5);
    // Contrast of a point = how far its color holds before the blend starts.
    let s = smoothstep(0.5 * p.point_1_contrast, 1.0 - 0.5 * p.point_2_contrast, t);
    store(c, vec4f(mix(p.point_1_color.rgb, p.point_2_color.rgb, s), 1.0));
}
