@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var f0 = ior_f0(p.ior);
    if (p.specular_f0 < 11) {
        let r = dielectric_f0_range(p.specular_f0);
        f0 = mix(r.x, r.y, select(p.specular_range, p.specular_range_glass, p.specular_f0 == 10));  // Glass has its own
    }
    store(c, vec4f(lin2srgb(vec3f(f0)), 1.0));
}
