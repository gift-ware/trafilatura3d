// Where rust grows: occlusion and edges, damaged varnish (cells), a noise; spread by Rust Spreading.
fn fx(c: vec2i) {
    let uv = mx_uv(c);
    let occ = 1.0 - load_baked_ambient_occlusion(c).r;
    let edge = abs(load_curvature(c).r - 0.5) * 2.0;
    let n = 4 + i32(p.varnish_damage_scale * 28.0);
    let dmg = smoothstep(0.35, 0.0, mx_cells(uv, vec2i(n), 1.0, p.seed_ + 7.0).x) * step(0.55, mx_cells(uv, vec2i(n), 1.0, p.seed_ + 7.0).z);
    let noise = mx_fbm(uv, 6, 5, 0.6, p.seed_ + 1.0);
    let x = occ * 1.2 + edge * 0.6 + dmg * 0.5 + (noise - 0.5) * 0.9 + 0.1;
    let r = mx_scan(x, p.rust_spreading, 1.0 - p.spreading_smoothness * 0.9);
    store_rust(c, vec4f(vec3f(r), 1.0));
}
