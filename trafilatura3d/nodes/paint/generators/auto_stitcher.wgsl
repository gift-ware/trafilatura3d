fn inside_uvs(c: vec2i) -> bool { return load_uv_islands(c).a >= 0.5; }

fn effect(c: vec2i) -> vec4f {
    let me = vec2f(c);
    if (p.stage_ == 0) {  // the region whose borders the paths follow
        var r = load_custom_input(c).r;
        if (p.mask_mode == 1) { r = select(0.0, 1.0, load_curvature(c).r >= 0.6); }
        return vec4f(vec3f(r), 1.0);
    }
    if (p.stage_ == 1) {  // distance to the region's border, in output pixels (0 outside the UVs in UV Mask mode)
        let s = load_seeds_(c);
        var d = 0.0;
        if (p.mask_mode == 0) {
            if (inside_uvs(c) && s.x >= 0.0) { d = distance(me, s.xy); }
        } else {
            let white = s.x >= 0.0 && all(s.xy == me);
            let other = select(s.xy, s.zw, white);  // nearest pixel of the other side
            if (other.x >= 0.0) { d = distance(me, other) - 0.5; }
        }
        return vec4f(vec3f(d), 1.0);
    }
    let pp = p.path_position * p.size_.x / 256.0;
    if (p.stage_ == 2) {  // path pixels: on the far side of the Path Position isoline with a neighbour before it
        let uv = uv_of(c);
        let t = 1.0 / p.size_;
        let d = sample_dist_(uv).r;
        let dx = sample_dist_(uv + vec2f(t.x, 0.0)).r - sample_dist_(uv - vec2f(t.x, 0.0)).r;
        let dy = sample_dist_(uv + vec2f(0.0, t.y)).r - sample_dist_(uv - vec2f(0.0, t.y)).r;
        let lo = min(min(sample_dist_(uv + vec2f(t.x, 0.0)).r, sample_dist_(uv - vec2f(t.x, 0.0)).r),
                     min(sample_dist_(uv + vec2f(0.0, t.y)).r, sample_dist_(uv - vec2f(0.0, t.y)).r));
        let valid = p.mask_mode != 0 || inside_uvs(c);
        let on = valid && d >= pp && lo < pp && length(vec2f(dx, dy)) > 1e-4;
        let tangent = normalize(vec2f(-dy, dx) + vec2f(1e-9, 0.0));
        return vec4f(select(0.0, 1.0, on), tangent, 1.0);
    }
    // Stitches around the nearest path pixel: along = its arc length (+ the offset along its tangent).
    let s = load_near_(c);
    if (s.x < 0.0 || (p.mask_mode == 0 && !inside_uvs(c))) { return vec4f(0.0, 0.0, 0.0, 1.0); }
    let b = vec2i(s.xy);
    let tb = texel_path_(b).gb;
    let o = me - s.xy;
    let along = texel_rank_(b).z - dot(o, tb);
    let across = dot(o, vec2f(-tb.y, tb.x));
    let period = max(p.stitch_size * 6.0 * p.size_.x / 256.0, 2.0);
    let k = floor(along / period);
    var q = vec2f(along - (k + 0.5) * period, across);
    q = g_rot(q, (hash21(vec2i(i32(k), 3), p.seed_) - 0.5) * p.jitter * 0.25);
    let half_ = vec2f(max(p.stitch_length * period * 0.5, 0.5), max(p.stitch_width * period * 0.5, 0.5));
    let e = mix(8.0, 2.0, p.stitch_roundness);
    let r = pow(pow(abs(q.x) / half_.x, e) + pow(abs(q.y) / half_.y, e), 1.0 / e);
    let v = sqrt(max(1.0 - r * r, 0.0));
    return vec4f(vec3f(v), 1.0);
}
