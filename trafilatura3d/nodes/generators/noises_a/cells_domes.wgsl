// Cells 1: an FX-Map walking the Scale grid, on every site (cells_sites.wgsl) one
// pattern (shapes.wgsl) 3 x Pattern size x Pattern scale cells wide (3.25 with the sites' 0.5 jitter would
// leave none of the preview's dark gaps), turned by Angle +- Angle random / 2, overlaid with Max, black where none
// reaches. Luminance random darkens each pattern by up to that much.

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let asp = aspect(p.non_square_expansion);
    let per = vec2i(round(f32(max(p.scale, 1)) * asp));
    let x = (uv_of(c) * asp + p.tile_offset) * vec2f(per) / asp;
    var kinds = array<i32, 7>(2, 3, 4, 5, 10, 14, 16);  // the Pattern list as shapes.wgsl kinds
    let v = cells_domes_at(x, per, kinds[clamp(p.pattern, 0, 6)], p.pattern_size * p.pattern_scale * 1.5, p.angle,
                       p.angle_random, p.luminance_random, p.disorder * p.disorder_speed, p.disorder_anisotropy,
                       p.disorder_anisotropy_angle);
    store(c, vec4f(v, v, v, 1.0));
}
