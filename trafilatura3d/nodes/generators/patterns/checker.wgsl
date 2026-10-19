@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = frame(c, p.non_square_expansion);
    let n = 2.0 * f32(max(p.tiling, 1));
    let g = grid(uv, n, p.rotate_45_degrees);
    let par = 1.0 - imod2(floor(g.x) + floor(g.y));          // white square at the top-left corner
    let f = fract(g);
    let dist = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));   // to the nearest square border
    let v = mix(0.5, par, clamp(2.0 * dist / grid_px(n, p.rotate_45_degrees), 0.0, 1.0));
    store(c, vec4f(v, v, v, 1.0));
}
