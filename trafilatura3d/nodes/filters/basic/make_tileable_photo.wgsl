// 1 at the edge, 0 from `size` (fraction of half the image) inwards; e = 0 at the edge, 1 at the centre.
fn edge_mask(e: f32, size: f32, precision_: f32, wobble: f32) -> f32 {
    let s = clamp(size, 1e-4, 1.0);
    let x = e + wobble * s;
    return 1.0 - smoothstep(s * clamp(precision_, 0.0, 0.999), s, x);
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let wh = (fbm(vec2f(0.0, uv.y), vec2i(8), p.seed_ + 3.0) - 0.5) * p.mask_warping_h / 100.0;
    let wv = (fbm(vec2f(uv.x, 0.0), vec2i(8), p.seed_ + 9.0) - 0.5) * p.mask_warping_v / 100.0;
    let h = edge_mask(1.0 - abs(2.0 * uv.x - 1.0), p.mask_size_h, p.mask_precision_h, wh);
    let v = edge_mask(1.0 - abs(2.0 * uv.y - 1.0), p.mask_size_v, p.mask_precision_v, wv);
    // Sampled with wrap-around whatever the node's tiling: the copies are the input moved by half.
    let s = vec2f(size_input());
    let a = texel_input(vec2i(floor(uv * s)));
    let b = texel_input(wrap(vec2i(floor((uv + vec2f(0.5, 0.0)) * s)), size_input()));
    let cc = texel_input(wrap(vec2i(floor((uv + vec2f(0.0, 0.5)) * s)), size_input()));
    let dd = texel_input(wrap(vec2i(floor((uv + vec2f(0.5, 0.5)) * s)), size_input()));
    store(c, mix(mix(a, b, h), mix(cc, dd, h), v));
}
