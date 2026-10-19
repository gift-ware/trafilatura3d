@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let v = load_basecolor(c);
    var metal = load_metallic(c).r;
    if (p.pbr_workflow == 1) { metal = 0.0; }
    let lin = srgb2lin(v.rgb);
    let y = luma(lin);
    let t = sat(p.tolerance);
    // Linear luminance limits: 50 / 30 sRGB = 0.0319 / 0.0122; 240 / 255 sRGB = 0.871 / 1.
    let lo = mix(mix(0.0319, 0.0122, t), mix(0.7, 0.5, t), metal);
    let hi = mix(0.871, 1.0, t);
    let target_y = clamp(y, lo, hi);
    var o = lin * (target_y / max(y, 1e-4));
    if (y < 1e-4) { o = vec3f(target_y); }
    store(c, vec4f(lin2srgb(o), v.a));
}
