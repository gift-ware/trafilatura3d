@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var p0 = p.target_position.rgb;
    var n0 = normalize(p.target_normal.rgb * 2.0 - 1.0 + vec3f(0.0, 0.0, 1e-5));
    if (p.project_input == 0) {
        p0 = sample_position_map(p.target_uv_position).rgb;
        n0 = normalize(sample_world_space_normal(p.target_uv_position).rgb * 2.0 - 1.0 + vec3f(0.0, 0.0, 1e-5));
    }
    let f = frame(n0);
    let r = rot_axis(n0, p.rotation * TAU);
    let t = r * f[0];
    let b = r * f[1];
    let pos = load_position_map(c).rgb;
    let n = normalize(load_world_space_normal(c).rgb * 2.0 - 1.0 + vec3f(0.0, 0.0, 1e-5));
    let half = max(p.scale * p.size, vec2f(1e-4)) * 0.5;
    let q = vec2f(dot(pos - p0, t), -dot(pos - p0, b)) / half;
    let inside = select(0.0, 1.0, abs(q.x) <= 1.0 && abs(q.y) <= 1.0);
    let depth = abs(dot(pos - p0, n0));
    let md = max(p.maximum_depth, 1e-4);
    let dm = 1.0 - smoothstep(md * (1.0 - p.depth_fade), md + 1e-4, depth);
    let al = dot(n, n0);
    let nm = smoothstep(p.normal_threshold - p.normal_fade * 0.5 - 1e-4, p.normal_threshold + p.normal_fade * 0.5, al);
    let m = inside * dm * nm;
    let v = sample_projected_texture(q * 0.5 + 0.5);
    store_output(c, vec4f(v.rgb * m, m));
    store_mask(c, g4(m));
}
