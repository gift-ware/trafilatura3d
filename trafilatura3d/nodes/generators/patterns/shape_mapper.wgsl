@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    // Centred coordinates, 1.0 = half the shortest side (or half of each side without expansion).
    var pos = ((vec2f(c) + 0.5) - p.size_ * 0.5) / (0.5 * p.size_);
    if (p.non_square_expansion == 1) { pos = ((vec2f(c) + 0.5) - p.size_ * 0.5) / (0.5 * min(p.size_.x, p.size_.y)); }
    let n = f32(max(p.pattern_amount, 1));
    // Angle clockwise from the right on screen (y down), minus the rotation, in turns.
    let ang = fract(atan2(pos.y, pos.x) / TAU - p.rotation);
    var t = ang;          // position along the outline, 0-1
    var dist = length(pos); // distance from the centre measured across the band
    var outer = p.radius;
    if (p.shape == 1) {
        var s = max(p.segments, 3);
        if (p.link_segments_with_pattern_amount == 1) { s = max(p.pattern_amount, 3); }
        let fs = f32(s);
        let k = min(floor(ang * fs), fs - 1.0);
        // Vertices of edge k (circumradius = Radius), in the rotated frame.
        let a0 = (k / fs + p.rotation) * TAU;
        let a1 = ((k + 1.0) / fs + p.rotation) * TAU;
        let v0 = vec2f(cos(a0), sin(a0));
        let v1 = vec2f(cos(a1), sin(a1));
        let e = v1 - v0;
        let nrm = normalize(vec2f(e.y, -e.x)) * select(1.0, -1.0, dot(vec2f(e.y, -e.x), v0 + v1) < 0.0);
        // Along the edge: projection of the point onto the edge direction (central projection).
        let rel = clamp(dot(pos - v0 * p.radius, e) / max(dot(e, e) * p.radius, 1e-6), 0.0, 1.0);
        t = (k + rel) / fs;
        dist = dot(pos, nrm);
        outer = p.radius * cos(PI / fs);
    }
    let w = max(p.width, 1e-5);
    var v = (outer + 0.5 * w - dist) / w;
    var u = t * n;
    let idx = i32(floor(u));
    u = fract(u);
    if (p.flip_one_on_two == 1 && (idx & 1) == 1) { v = 1.0 - v; }
    var o = 0.0;
    if (v >= 0.0 && v <= 1.0) {
        let uv = vec2f(u, v);
        if (p.filtering_mode == 0) {
            let s = vec2f(size_input());
            o = texel_input(vec2i(floor(uv * s))).r;
        } else {
            o = sample_input(uv).r;
        }
    }
    store(c, vec4f(o, o, o, 1.0));
}
