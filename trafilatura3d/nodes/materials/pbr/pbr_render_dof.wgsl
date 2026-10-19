// Depth of field by gathering: polygonal aperture (blades), ring, chromatic diffraction, swirl,
// or the custom aperture map. Depth is in the alpha of the rendered image.
fn focus_depth() -> f32 {
    let d = vec2f(size());
    var q = vec2f(0.5);
    if (p.focus_mode == 1) { q = p.focus_point; }
    var z = load_hdr(vec2i(q * d)).a;
    if (z > 1e3) { z = p.distance; }
    return max(z + p.focus_offset * p.distance, 0.01);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let fz = focus_depth();
    let z = min(load_hdr(c).a, 100.0);
    let coc = min(p.aperture_radius * 6.0 * abs(z - fz) / max(z, 0.01), 0.08);  // radius in UV
    if (coc < 0.5 / f32(d.x)) { store(c, vec4f(load_hdr(c).rgb, 1.0)); return; }
    let nb = f32(max(p.aperture_blades, 3));
    let swirl = p.swirly_bokeh * length(uv - 0.5) * 3.0;
    var acc = vec3f(0.0);
    var wsum = vec3f(0.0);
    let n = 64;
    for (var i = 0; i < n; i++) {
        let r = sqrt((f32(i) + 0.5) / f32(n));
        let a = f32(i) * 2.39996323;
        // Polygon: the disc scaled to the blade edges.
        let seg = TAU / nb;
        let poly = cos(PI / nb) / cos((a - seg * floor(a / seg)) - PI / nb);
        var o = vec2f(cos(a), sin(a)) * r * poly;
        var w = mix(1.0, r * r * 2.0, p.aperture_ring);
        if (p.use_custom_aperture_map != 0) { w = sample_lens_aperture_map(vec2f(cos(a), sin(a)) * r * 0.5 + 0.5).r; }
        o = rot2(o, swirl * r);
        let k = vec3f(1.0 + p.aperture_diffraction * 0.04, 1.0, 1.0 - p.aperture_diffraction * 0.04);
        let sr = sample_hdr(uv + o * coc * k.r).r;
        let sg = sample_hdr(uv + o * coc).g;
        let sb = sample_hdr(uv + o * coc * k.b).b;
        acc += vec3f(sr, sg, sb) * w;
        wsum += vec3f(w);
    }
    store(c, vec4f(acc / max(wsum, vec3f(1e-4)), 1.0));
}
