// The FX-Map Roughness weighting (fxmap.ROUGHNESS_WGSL): the weight of level k (counted from Min Level) of n,
// x Global Opacity.
fn level_weight(k: i32, n: i32) -> f32 {
    return roughness_weight(p.roughness, f32(k), f32(n)) * p.global_opacity;
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let uv = expand_uv(pix_uv(c), p.non_square_expansion);
    let lo = clamp(min(p.min_level, p.max_level), 0, 4);
    let hi = clamp(max(p.min_level, p.max_level), 0, 4);
    var acc = 0.0;
    for (var l = lo; l <= hi; l++) {
        var cells = max(p.scale * p.size * pow(max(p.lacunarity, 1.0), f32(l)), vec3f(1e-3));
        var per = vec3i(1 << 20);
        if (p.enable_tiling != 0) {
            cells = max(round(cells), vec3f(1.0));
            per = vec3i(cells);
        }
        let s = p.seed_ + f32(l) * 19.3;
        let v = vstyle(voronoi3(vpoint(uv, cells), per, s, color_seed() + f32(l) * 5.0));
        if (p.blend_mode == 0) {
            acc += level_weight(l - lo, hi - lo + 1) * v;
        } else if (p.blend_mode == 1) {
            acc = select(max(acc, level_weight(l - lo, hi - lo + 1) * v), level_weight(0, hi - lo + 1) * v, l == lo);
        } else {
            // Min: no Roughness weighting (every level weighs Global Opacity).
            acc = select(min(acc, v), v, l == lo);
        }
    }
    if (p.blend_mode == 2) { acc *= p.global_opacity; }
    var v = sat(acc);
    if (p.invert != 0) { v = 1.0 - v; }
    store(c, vec4f(v, v, v, 1.0));
}
