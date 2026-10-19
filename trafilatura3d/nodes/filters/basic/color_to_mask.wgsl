// Distance from the pixel's colour to the reference, normalised to 0..1 by the largest distance the space allows.
fn mask_distance(a: vec3f, b: vec3f) -> f32 {
    switch p.distance_color_space {
        case 1: { return distance(srgb_to_linear(a), srgb_to_linear(b)) / length(vec3f(1.0)); }  // cube diagonal
        case 2: {
            // Michelson contrast of the two Rec. 601 lumas: |La - Lb| / (La + Lb).
            let l = vec2f(dot(a, vec3f(0.299, 0.587, 0.114)), dot(b, vec3f(0.299, 0.587, 0.114)));
            return abs(l.x - l.y) / max(l.x + l.y, 1e-6);
        }
        case 3: { return delta_e2000(rgb_to_lab(a), rgb_to_lab(b), p.distance_weights) / 100.0; }
        case 4: {
            // Angle between the decoded normals, as a fraction of a half turn.
            let na = normalize(2.0 * a - 1.0 + vec3f(0.0, 0.0, 2e-6));
            let nb = normalize(2.0 * b - 1.0 + vec3f(0.0, 0.0, 2e-6));
            return acos(clamp(dot(na, nb), -1.0, 1.0)) / PI;
        }
        default: { return distance(a, b) / length(vec3f(1.0)); }
    }
}

@compute @workgroup_size(WG, WG)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let d = size();
    let c = vec2i(id.xy);
    if (c.x >= d.x || c.y >= d.y) { return; }
    let ci = load_color_input(c).rgb;
    let b = min(select(p.color, ci, p.use_color_input != 0), vec3f(1.0));
    let dist = mask_distance(min(load_input(c).rgb, vec3f(1.0)), b);
    // White up to (1 - fade) x Threshold, a linear fall to black at Threshold (fade 0: a hard cut).
    let t = max(p.threshold, 1e-6);
    let soft = p.selection_fade * t;
    var m = select(0.0, 1.0, dist < t);
    if (p.selection_fade > 1e-6) { m = clamp((t - dist) / soft, 0.0, 1.0); }
    store(c, gray4(m));
}
