// Ambient occlusion (Deprecated), ours: cavities = how far the height lies under its blur (`blurred`), darkened by
// p.depth per unit; the result raised to 1 + the facing (blue of the height's normal at slope scale p.steep, in 256
// px units): a cavity on flat ground counts twice, one on a steep wall once.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let h = load_height(c).r;
    let under = max(sample_blurred((vec2f(c) + 0.5) / vec2f(d)).r - h, 0.0);
    let k = p.steep * f32(d.x) / 512.0;  // central differences in 256 px units
    let g = vec2f(load_height(c + vec2i(1, 0)).r - load_height(c - vec2i(1, 0)).r,
                  load_height(c + vec2i(0, 1)).r - load_height(c - vec2i(0, 1)).r) * k;
    let facing = inverseSqrt(1.0 + dot(g, g));
    let ao = pow(clamp(1.0 - p.depth * under, 0.0, 1.0), 1.0 + facing);
    store(c, vec4f(ao, ao, ao, 1.0));
}
