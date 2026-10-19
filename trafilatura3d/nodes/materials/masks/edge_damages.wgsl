@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let e = convex(load_curvature(c).r) * mix(0.3, 1.0, load_ambient_occlusion(c).r);
    let ch = cells(uv, 24, 1.0, p.seed_);
    let chips = smoothstep(0.75, 0.2, ch.x) * (0.5 + 0.5 * ch.z);
    let scr = max(scratches(uv, 16, 1.5, 0.0, 2.0, 0.06, p.seed_ + 7.0), fbm(uv, 16, 4, 0.7, p.seed_ + 3.0) * 0.8);
    let n = mix(chips, scr, p.damages_intensity);
    store(c, g4(hscan(e * (0.4 + n), p.level, p.contrast) * load_mask(c).r));
}
