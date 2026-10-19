@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = vec2i(id.xy);
    if (c.x >= size().x || c.y >= size().y) { return; }
    let a = 6.28318531 * p.direction;
    let s = sin(6.28318531 * p.slope_angle);
    let y = select(-1.0, 1.0, p.normal_format == 1);  // DirectX: the green turned half a turn
    store(c, vec4f(0.5 - 0.5 * cos(a) * s, 0.5 + 0.5 * y * sin(a) * s, 1.0, 1.0));
}
