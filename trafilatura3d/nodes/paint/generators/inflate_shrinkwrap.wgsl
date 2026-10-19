fn effect(c: vec2i) -> vec4f {
    let depth = p.surface_depth_cm;
    let s = load_seeds_(c);
    var h = 0.0;
    if (load_uv_islands(c).a >= 0.5 && s.x >= 0.0) {
        let cm = p.surface_size_cm / p.size_.x;  // centimetres per output pixel
        let d = distance(vec2f(c), s.xy) * cm;
        // Profile rising from the seam over R.
        let r = max(depth * 3.0, 1e-3);
        let t = clamp(d / r, 0.0, 1.0);
        var prof: f32;
        switch p.preset {
            case 1: { prof = pow(t, 0.3); }                    // Vacuum Pulled: steep at the seam, then flat
            case 2: { prof = smoothstep(0.0, 0.3, t); }        // Tight: a short ramp
            default: { prof = sqrt(1.0 - (1.0 - t) * (1.0 - t)); }  // Inflated: round pillow
        }
        let cavity = clamp((0.5 - load_curvature(c).r) * 4.0, 0.0, 1.0);
        let shrink = p.inflate_or_shrinkwrap == 1;
        h = depth * select(prof, 0.5 + 0.2 * prof, shrink);
        // Seam groove and raised lip.
        let sw = max(depth * 0.15, cm);
        h -= p.seam_intensity * depth * 0.5 * exp(-(d * d) / (sw * sw));
        let lw = max(p.raised_edge_width * r * 0.5, cm);
        let lip = (d - lw) / (lw * 0.6);
        h += p.raised_edge_intensity * depth * 0.4 * exp(-lip * lip);
        // Wrinkles: ridges across the border, from the noise at the nearest border point.
        let range = max(p.wrinkle_range * 0.15 * p.surface_size_cm, cm);
        let fade = 1.0 - smoothstep(0.0, range, d);
        let freq = (10.0 + 150.0 * p.wrinkle_density) / (0.5 + p.wrinkle_scale);
        let pull = 1.0 + p.wrinkle_tightness * 2.0 * fade;
        let b = s.xy / p.size_.x;  // the border point in UV
        let ridge = g_ridged(vec3f(b * freq * pull, d / range * 0.7), p.seed_);
        var amp = depth * (0.2 + 0.8 * p.wrinkle_scale) * 0.4 * select(1.0, 2.0, shrink);
        if (shrink) { amp *= 0.5 + cavity; } else { h += depth * 0.2 * cavity * prof; }
        h += amp * fade * (ridge * ridge - 0.4);
    }
    let v = p.height_position + (h - depth * 0.5) / max(p.height_range, 1e-3);
    return vec4f(vec3f(clamp(v, 0.0, 1.0)), 1.0);
}
