@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let up = sat(wsn(load_world_space_normal(c).rgb).y);
    let occ = 1.0 - load_ambient_occlusion(c).r;
    var v = up * (1.0 - 0.5 * p.occlusion_amount) + occ * p.occlusion_amount * 0.8 * smoothstep(-0.2, 0.3, up);
    var n = fbm(uv, 6, 6, 0.6, p.seed_);
    if (p.override_noise != 0) { n = load_noise(c).r; }
    v = mix(v, v * n * 2.0, p.noise_opacity);
    store(c, g4(hscan(v, p.level, p.contrast) * load_mask(c).r));
}
