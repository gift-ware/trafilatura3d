@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    var u = unit_slice(c);
    if (POSITION_MAP_CONNECTED) { u = load_position_map(c).rgb; }
    let f = freq3(f32(max(p.scale, 1)), p.size, true);
    let r = voronoi3(u * f, per3(f, true), vec3f(1.0), p.mode, p.minkowski_number, p.seed_);
    if (p.style == 4) {
        var col = h33(r.id, p.seed_ + 9.0);
        if (p.invert != 0) { col = 1.0 - col; }
        store(c, vec4f(col, 1.0));
        return;
    }
    var v = r.f1;
    if (p.style == 1) { v = r.f2; }
    else if (p.style == 2) { v = r.f2 - r.f1; }
    v = clamp(v, 0.0, 1.0);
    if (p.style == 3) {
        let aa = max(f.x, f.y) / p.size_.x;
        v = 1.0 - smoothstep(p.border_width * 0.5 - aa, p.border_width * 0.5 + aa, r.f2 - r.f1);
    } else {
        // Roundness: the slope around each point becomes a round (convex) dome.
        v = mix(v, 1.0 - sqrt(max(1.0 - v * v, 0.0)), clamp(p.roundness, 0.0, 1.0));
    }
    if (p.invert != 0) { v = 1.0 - v; }
    store(c, vec4f(v, v, v, 1.0));
}
