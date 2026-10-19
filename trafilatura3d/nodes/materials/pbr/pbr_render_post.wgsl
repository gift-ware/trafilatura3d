@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_of(c, d);
    let ex = exp2(p.exposure);
    var col = sample_img(uv).rgb * ex;
    if (p.enable_post_effects != 0) {
        let ctr = uv - 0.5;
        let s = p.bloom_chroma_shift * 0.03;
        let bl = vec3f(sample_bloom(0.5 + ctr * (1.0 + s)).r, sample_bloom(uv).g, sample_bloom(0.5 + ctr * (1.0 - s)).b)
            + sample_bloom2(uv).rgb;
        let dirt = mix(1.0, sample_lens_dirt_map(uv).r * 2.0, p.lens_dirt_intensity);
        // Halo: a ring of the bloom seen through the opposite side; flares: ghosts mirrored about the centre.
        let hdir = normalize(-ctr + vec2f(1e-5)) * 0.4;
        let halo = sample_bloom2(fract(uv + hdir)).rgb * p.lens_halo_intensity * smoothstep(0.1, 0.4, length(ctr));
        var fl = vec3f(0.0);
        for (var k = 1; k <= 3; k++) {
            let g = 0.5 - ctr * (0.5 + f32(k) * 0.35);
            fl += sample_bloom2(clamp(g, vec2f(0.0), vec2f(1.0))).rgb * vec3f(1.0 - f32(k) * 0.2, 0.8, 0.6 + f32(k) * 0.15);
        }
        col += (bl * p.bloom_intensity + halo + fl * p.lens_flares_intensity * 0.5) * dirt;
    }
    var o: vec3f;
    switch p.tone_mapper {
        case 1: { o = lin2srgb(tonemap_aces(col)); }
        case 2: { o = tonemap_hejl(col); }
        default: { o = lin2srgb(sat3(col)); }
    }
    let r = length(uv - 0.5) * 1.4142;
    o *= 1.0 - p.vignette_intensity * smoothstep(p.vignette_radius, p.vignette_radius + 0.6, r);
    store(c, vec4f(o, 1.0));
}
