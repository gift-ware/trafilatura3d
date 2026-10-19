// Absolute value of a gradient noise on the Scale lattice: broad bright lobes split by long dark
// valleys where the noise crosses zero. Disorder turns the lattice gradients (phase shift).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let per = max(p.scale, 1);
    let v = sat(1.15 * abs(gnoise_ph(uv * f32(per), vec2i(per), p.seed_, p.disorder)));
    store(c, vec4f(v, v, v, 1.0));
}
