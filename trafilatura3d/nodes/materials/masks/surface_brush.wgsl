@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    // Streak direction varies slowly with the height (Position) so they follow the volume a bit.
    let ang = (load_position(c).r - 0.5) * 0.6;
    let len = 0.3 + p.scratches_length * 1.2;
    var s = scratches(uv, 32, len, ang, 0.08, 0.12, p.seed_);
    s = max(s, scratches(uv, 64, len * 0.7, ang, 0.08, 0.14, p.seed_ + 3.0) * 0.7);
    let n = wsn(load_world_space_normal(c).rgb);
    if (p.occlude_axis < 3) {
        let a = abs(n[p.occlude_axis]);
        s *= mix(1.0, a, p.occlude_axis_intensity);
    }
    s *= mix(1.0, load_ambient_occlusion(c).r, p.occlusion);
    s *= 1.0 - 0.7 * convex(load_curvature(c).r);
    s = sat((s - 0.5) * (1.0 + p.sharpen_intensity * 3.0) + 0.5);
    store(c, g4(hscan(s, p.level, p.contrast) * load_mask(c).r));
}
