@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let dir = ll_dir(uv_of(c, d));
    let t = dot(dir, ll_dir(p.point_position)) * 0.5 + 0.5;
    let k = sat((t - 0.5) / max(1.0 - p.contrast, 0.002) + 0.5);
    store(c, vec4f(mix(p.bottom_color.rgb, p.top_color.rgb, k), 1.0));
}
