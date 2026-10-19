// Weave Twill (Legacy)'s Disorder, ours: one texel per cell of the 16 x 16 lattice, the moves of its two floats (xy:
// the first family's, zw: the second's), `disorder` cells each at a random angle.

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= 16 || c.y >= 16) { return; }
    let a = 6.28318531 * hash22(c, p.seed_);
    store(c, p.disorder * vec4f(cos(a.x), sin(a.x), cos(a.y), sin(a.y)));
}
