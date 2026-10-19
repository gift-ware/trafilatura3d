// The up axis of the position map (Position axis: Y-Green, X-Red, B-Blue) as a channel index.
fn axis_index() -> i32 { return select(select(2, 0, p.position_axis == 1), 1, p.position_axis == 0); }
fn pos_at(q: vec2i) -> vec3f { return texel_position(clamp(q, vec2i(0), size_position() - 1)).rgb; }

fn effect(c: vec2i) -> vec4f {
    let uv = uv_of(c);
    if (p.stage_ == 0) {  // where water stays: cavities and occlusion
        let curv = g_invert(load_curvature(c).r, p.invert);
        let ao = g_invert(load_baked_ambient_occlusion(c).r, p.invert);
        let s = max(clamp((0.5 - curv) * 3.0, 0.0, 1.0), clamp((0.8 - ao) * 2.5, 0.0, 1.0));
        return vec4f(vec3f(s), 1.0);
    }
    if (p.stage_ == 1) {  // spread rust, broken by a world space grunge
        // The blurred sources thresholded lower with more spreading: the rust grows around them.
        let s = sample_spread_(uv).r;
        let thr = mix(0.5, 0.04, p.rust_spreading);
        let w = thr * (0.1 + 0.9 * p.spreading_smoothness);
        let g = fbm(load_position(c).rgb * 12.0, p.seed_ + 3.0);
        let grown = smoothstep(thr - w, thr + w, s) * smoothstep(0.2, 0.6, g + s);
        let r = max(load_sources_(c).r, grown);
        return vec4f(vec3f(r), 1.0);
    }
    if (p.stage_ == 2) {  // drips: march uphill and take the rust found above, fading with the distance
        let a = axis_index();
        let qc = vec2i(floor(uv * vec2f(size_position())));
        let k_px = f32(size_position().x) / p.size_.x;  // position texels per output pixel
        let jx = (pos_at(qc + vec2i(1, 0)) - pos_at(qc - vec2i(1, 0))) * 0.5 * k_px;  // dP per output pixel
        let jy = (pos_at(qc + vec2i(0, 1)) - pos_at(qc - vec2i(0, 1))) * 0.5 * k_px;
        let grad = vec2f(jx[a], jy[a]);
        let gl = length(grad);
        var drip = 0.0;
        if (gl > 1e-7) {
            let dir = grad / gl;  // uphill, in output pixels
            let n = max(p.drips_samples_amount, 1);
            let step_px = p.drips_intensity * 0.25 * p.size_.x / f32(n);
            var prev = load_position(c).rgb;
            // A step moves the position by about |dP/ds| x step: much more is a jump to another island.
            let jump = 3.0 * length(jx * dir.x + jy * dir.y) * step_px + 2e-3;
            for (var k = 1; k <= n; k++) {
                let q = vec2f(c) + 0.5 + dir * step_px * f32(k);
                if (any(q < vec2f(0.0)) || any(q >= p.size_)) { break; }
                let quv = q / p.size_;
                let pq = sample_position(quv).rgb;
                if (distance(pq, prev) > jump) { break; }  // a UV seam
                prev = pq;
                let fall = 1.0 - f32(k) / f32(n);
                drip = max(drip, sample_rust_(quv).r * fall * sqrt(fall));
            }
            // Streaks: noise varying fast across the drips (the two other axes), slowly along them.
            let pc = load_position(c).rgb;
            var h = vec2f(pc.x, pc.z);
            if (a == 0) { h = pc.yz; } else if (a == 2) { h = pc.xy; }
            let streak = smoothstep(0.35, 0.65, fbm(vec3f(h * 70.0, pc[a] * 4.0), p.seed_ + 9.0));
            drip *= streak;
        }
        return vec4f(vec3f(drip), 1.0);
    }
    let x = max(load_rust_(c).r, sample_drips_(uv).r);
    return vec4f(vec3f(g_contrast(x, p.rust_contrast)), 1.0);
}
