// Linear falloff from a shape's edge: 1 on the shape, 0 at `reach` pixels from it, on the glowing side (outside
// when `outer`, else inside the shape).
@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let sd = shape_sd(c);
    let x = select(-sd, sd, p.outer == 1);
    store_out(c, fx_gray(clamp(1.0 - max(x, 0.0) / max(p.reach, 1e-3), 0.0, 1.0)));
}
