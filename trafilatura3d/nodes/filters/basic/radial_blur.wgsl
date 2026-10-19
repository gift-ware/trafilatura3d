@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let uv = uv_at(c);
    let aspect = vec2f(1.0, p.size_.y / p.size_.x);
    let centre = 0.5 + clamp(p.center_position, vec2f(-0.5), vec2f(0.5));
    let v = (uv - centre) * aspect;
    let n = clamp(p.samples, 1, 128);
    var acc = vec4f(0.0);
    for (var k = 0; k < n; k++) {
        acc += sample_input(centre + rot(v, p.angle * f32(k) / f32(n)) / aspect);
    }
    store(c, acc / f32(n));
}
