@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let place = convex(load_curvature(c).r) * 0.5 + (1.0 - load_ambient_occlusion(c).r) * 0.5;
    let n = grunge(uv, 6.0, p.seed_) * 0.7 + fbm(uv, 24, 3, 0.7, p.seed_ + 1.0) * 0.4;
    store(c, g4(hscan(n * 0.6 + place * 0.6, p.rust_spreading, p.rust_contrast)));
}
