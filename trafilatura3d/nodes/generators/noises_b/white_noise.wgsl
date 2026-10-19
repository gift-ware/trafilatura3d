// Per pixel: Disorder shifts each pixel's uniform values around the unit loop (animation keeps the
// histogram); Gaussian = Box-Muller (sigma 1/6 around 0.5), Triangle = mean of two uniforms.
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let shift = p.disorder * p.disorder_speed;
    let u1 = fract(hash21(c, p.seed_) + shift);
    let u2 = fract(hash21(c, p.seed_ + 17.0) + shift);
    var v = u1;
    if (p.noise_distribution == 1) {
        let g = sqrt(-2.0 * log(max(u1, 1e-7))) * cos(TAU * u2);
        v = sat(0.5 + g / 6.0);
    } else if (p.noise_distribution == 2) {
        v = (u1 + u2) * 0.5;
    }
    store(c, vec4f(v, v, v, 1.0));
}
