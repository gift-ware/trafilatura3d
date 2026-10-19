// Advection: walk along the field in 16 steps, so content flows along curved vectors; reads
// outside the canvas are clamped to its edge (no wrap-around).
fn inside(q: vec2f) -> vec2f { return clamp(q, 0.5 / p.size_, 1.0 - 0.5 / p.size_); }

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var q = fx_uv(c);
    for (var i = 0; i < 16; i++) {
        q += (sample_vector_field(inside(q)).rg - 0.5) * p.amount / 16.0;
    }
    store(c, sample_input(inside(q)));
}
