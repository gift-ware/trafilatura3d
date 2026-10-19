@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    let wv = max(p.waves, 1);
    // Layers: the band coordinate t, bent by Waves sine waves whose phase and amplitude are turned
    // randomly per wave (smoothly), folded hard by a warped fbm (Distortion: the layers break and
    // meander), with fine Details. The bands are thin dark seams between bright folded layers.
    let rr = p.rotation_random;
    let ph = rr * (vnoise(vec2f(uv.x * f32(wv), uv.y * 3.0), vec2i(wv, 3), s + 3.0) - 0.5) * 2.0;
    let amp = 0.06 * (1.0 + rr * (vnoise(vec2f(uv.x * f32(wv), uv.y * 2.0), vec2i(wv, 2), s + 4.0) - 0.5) * 3.0);
    let wq = uv + p.distortion * 0.3 * warp_vec(uv, 3, 4, s + 5.0);
    var t = wq.y + amp * sin(TAU * (f32(wv) * wq.x + ph));
    t += p.distortion * 1.6 * gfbm(wq, 2, 5, 0.55, s + 7.0);
    t += p.details * 0.03 * gfbm(uv, 16, 3, 0.6, s + 9.0);
    let bands = fract(t * f32(max(p.divisions, 1)));
    let d = min(bands, 1.0 - bands) * 2.0;  // 0 on a seam, 1 between two
    var v = 0.03 + 0.97 * smoothstep(0.0, 0.4, d) * mix(0.6, 1.0, smoothstep(0.0, 0.7, bands));
    v -= 0.25 * smoothstep(0.5, 0.8, fbm(uv, 6, 3, 0.5, s + 11.0)) * (1.0 - d);
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
