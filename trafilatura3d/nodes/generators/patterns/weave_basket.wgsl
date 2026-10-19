@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let t = f32(max(p.tiling, 1));
    let n = 4.0 * exp2(select(min(floor(sqrt(t)), 3.0), 0.0, t == 1.0));
    let g = grid(uv, n, p.rotate_45_degrees);
    // Basket weave: blocks of five parallel capsule threads, vertical where i + j is even,
    // horizontal elsewhere.
    let b = floor(g);
    var f = fract(g);
    if (imod2(b.x + b.y) == 1.0) { f = f.yx; }
    let k = clamp(floor(f.x * 5.0), 0.0, 4.0);
    let v = capsule(vec2f(f.y - 0.5, f.x - (k + 0.5) / 5.0), 0.3877, vec2f(0.1073, 0.0820), 2.18);
    store(c, vec4f(v, v, v, 1.0));
}
