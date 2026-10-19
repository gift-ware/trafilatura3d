@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let s = p.seed_;
    // Small crisp splattered spots on two scales; Details warps their outlines and breaks some into
    // finer droplets; a soft coverage mask gathers them.
    let q = uv + p.details * 0.008 * warp_vec(uv, 96, 2, s + 2.0);
    let sp = max(discs(q, 48, 1, 0.2, 0.1, 0.32, 0.15, s).x, discs(q, 128, 1, 0.12, 0.15, 0.4, 0.2, s + 3.0).x);
    let brk = mix(1.0, smoothstep(0.3, 0.45, fbm(q, 256, 2, 0.5, s + 6.0)), 0.5 * p.details);
    let k = mix(0.25, 0.02, sat(p.coverage_contrast));
    let cov = smoothstep(1.0 - p.coverage - k, 1.0 - p.coverage + k, fbm(uv, 6, 4, 0.5, s + 8.0));
    var v = smoothstep(0.3, 0.6, sp * brk) * cov;
    store(c, vec4f(sat(v), sat(v), sat(v), 1.0));
}
